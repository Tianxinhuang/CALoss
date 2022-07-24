'''
Created on September 2, 2017

@author: optas
'''
import numpy as np
import tensorflow as tf
import random
from encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only,conv2d,fully_connect,batch_normalization,get_direction,decoder_with_folding_only
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
from pointnet_util import pointnet_sa_module_msg,pointnet_sa_module
def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        bnum=tf.shape(xyz)[0]
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        #new_xyz=tf_sampling.gather_point(xyz,idx)
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=np.arange(ptnum)
        ptids=tf.random_shuffle(ptids,seed=None)
        #random.shuffle(ptids)
        #print(ptids,ptnum,npoint)
        #ptidsc=ptids[tf.py_func(np.random.choice(ptnum,npoint,replace=False),tf.int32)]
        ptidsc=ptids[:npoint]
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
        #ptid=tf.tile(tf.constant(ptidsc,shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])
        
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
def global_fix(scope,cens,feats,mlp=[128,128],mlp1=[128,128]):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        tensor0=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
        #tensor0=tf.expand_dims(tf.concat([cens,feats,tf.tile(gfeat,[1,tf.shape(feats)[1],1])],axis=-1),axis=2)
        tensor=tensor0
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensorword=tensor
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=tf.concat([tf.expand_dims(cens,axis=2),tf.expand_dims(feats,axis=2),tf.tile(tensor,[1,tf.shape(feats)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp1):
            tensor=conv2d('global_ptstate2%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensor=conv2d('global_ptout',tensor,3+feats.get_shape()[-1],[1,1],padding='VALID',activation_func=None)
        newcens=cens#+tf.squeeze(tensor[:,:,:,:3],[2])
        #newcens=tf.nn.tanh(newcens)
        newfeats=feats+tf.squeeze(tensor[:,:,:,3:],[2])
    tf.add_to_collection('cenex',tf.reduce_mean(tf.abs(tensor[:,:,:,:3])))
        #tensor=tf.expand_dims(tf.concat([feats,gfeat],axis=-1),axis=2)
        #for i,outchannel in enumerate(mlp1):
        #    tensor=conv2d('global_feat%d'%i,tensor,outchannel,[1,1],padding='VALID')
        #tensor=conv2d('nonlocal_featout',tensor,feats.get_shape()[-1].value,[1,1],padding='VALID',activation_func=None)
    return newcens,newfeats
def local_kernel(l0_xyz,cenlist=None,pooling='max',it=True):
    l0_points=None
    if cenlist is None:
        cen11,cen22=None,None
    else:
        cen11,cen22=cenlist
    cen1,feat1=pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.2], [16], [[32,32,64]],cens=cen11, is_training=it, bn_decay=None, scope='layer1', use_nchw=False,bn=True,use_knn=True,pooling=pooling)
    cen2,feat2=pointnet_sa_module_msg(cen1, feat1, 128, [0.4], [16], [[64,64,128]],cens=cen22, is_training=it, bn_decay=None, scope='layer2',use_nchw=False,bn=True,use_knn=True,pooling=pooling)
    l3_xyz, rfeat3,_ = pointnet_sa_module(cen2, feat2, npoint=None, radius=None, nsample=None, mlp=[128,256,128], mlp2=None, group_all=True, is_training=it, bn_decay=None, scope='layer3',pooling=pooling)
    #cen3,feat3=pointnet_sa_module_msg(cen2, feat2, 32, [0.6], [16], [[128,128,256]], is_training=it, bn_decay=None, scope='layer3',use_nchw=False,bn=True,use_knn=False,pooling=pooling)
    #rcen3,rfeat3=global_fix('global3',cen3,feat3,mlp=[256,256],mlp1=[256,256])
    #return [cen1,cen2],feat2
    #return rcen3,rfeat3
    return [cen1,cen2],tf.squeeze(rfeat3,[1])
def global_kernel(xyz,pooling='max',mlps=[64, 128, 128, 256, 128],acti_func=tf.nn.relu,secword=None):
    feat,mfeat=adaptive_loss_net(xyz,n_filter=mlps,activation_func=acti_func,normal=False,pooling=pooling,secword=secword)
    return feat,mfeat
def movenet(inpts,startcen0,knum=64,mlp1=[128,128],mlp2=[128,128]):
    #with tf.variable_scope(scope):
    ptnum=inpts.get_shape()[1].value
    #_,startcen=sampling(knum,inpts,use_type='r')
    #_,allcen=sampling(ptnum,inpts,use_type='f')
    #_,startcen=sampling(knum,allcen,use_type='n')
    #_,startcen=sampling(knum,inpts,use_type='r')
    startcen=startcen0[:,:knum]
    #startcen=inpts
    words=tf.expand_dims(startcen,axis=2)
    inwords=tf.expand_dims(inpts,axis=2)

    for i,outchannel in enumerate(mlp1):
        inwords=conv2d('movein_state%d'%i,inwords,outchannel,[1,1],padding='VALID',activation_func=None)
        inwords=tf.nn.leaky_relu(inwords)
    inwords=tf.reduce_sum(inwords,axis=1,keepdims=True)/ptnum
    #inwords=tf.reduce_mean(inwords,axis=1,keepdims=True)

    for i,outchannel in enumerate(mlp1):
        words=conv2d('mover_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
        words=tf.nn.leaky_relu(words)
    wordsfeat=words
    #words=tf.reduce_sum(words,axis=1,keepdims=True)/ptnum
    words=tf.reduce_mean(words,axis=1,keepdims=True)

    #words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1])],axis=-1)
    words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1]),tf.tile(inwords,[1,tf.shape(startcen)[1],1,1])],axis=-1)
    for i,outchannel in enumerate(mlp2):
        words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
        words=tf.nn.leaky_relu(words)
    words=conv2d('basic_stateoutg',words,3,[1,1],padding='VALID',activation_func=None)
    move=tf.squeeze(words,[2])
    result=startcen+move
    result=tf.concat([result,startcen0[:,knum:]],axis=1)
    movelen=tf.sqrt(tf.reduce_sum(tf.square(move),axis=-1))
    return result,movelen
def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, dnum=3, bneck_post_mlp=False,mode='fc'):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    #decoder = decoder_with_fc_only
    #decoder = decoder_with_folding_only

    n_input = [n_pc_points, dnum]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': False,
                    'non_linearity':tf.nn.relu
                    }
    if mode=='fc':
        decoder = decoder_with_fc_only
        decoder_args = {'layer_sizes': [256,256, np.prod(n_input)],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False
                        }
    else:
        decoder = decoder_with_folding_only
        decoder_args = {'layer_sizes': [256,256,dnum],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False
                        }
    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args
def adaptive_loss_net(input_signal,n_filter=[64,128,256],activation_func=tf.nn.relu,normal=False,pooling='max',secword=None):
    #with tf.variable_scope('ad',reuse=True):
    encoder = encoder_with_convs_and_symmetry
    enc_args = {'n_filters': n_filter,
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': False,
                    'verbose': True
                    }
    #words=encoder(input_signal,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],symmetry=tf.reduce_max,strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
    words=encoder(input_signal,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],symmetry=None,non_linearity=activation_func,strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
    #words=tf.expand_dims(input_signal,axis=2)
    #for i,outchannel in enumerate(n_filter[:-1]):
    #    words=conv2d('basic_state1%d'%i,words,outchannel,[1,1],padding='VALID')
    
    #words=tf.squeeze(conv2d('basic_stateout',tf.expand_dims(words,axis=2),n_filter[-1],[1,1],padding='VALID',activation_func=None),axis=2)
    #words=batch_normalization(words)
    #words=tf.squeeze(words,axis=[2])
    if pooling is 'max':
        word=tf.reduce_max(words,axis=1)
    elif pooling is 'maxmean':
        word=tf.reduce_max(words,axis=1,keepdims=True)
        meanword=tf.reduce_mean(words,axis=1,keepdims=True)
        minword=tf.reduce_min(words,axis=1,keepdims=True)
        if secword is None:
            #secword=tf.square(word-meanword)/2
            secword=tf.concat([word,meanword],axis=-1)
            secword=tf.expand_dims(secword,axis=1)
            for i,outchannel in enumerate([128,128]):
                secword=conv2d('sec_state%d'%i,secword,outchannel,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)
            secword=conv2d('sec_stateoutg',secword,word.get_shape()[-1].value,[1,1],padding='VALID',activation_func=None)
            #secword=conv2d('sec_stateoutg',secword,1,[1,1],padding='VALID',activation_func=None)
            secword=tf.square(secword)
            secword=(word-meanword)/(0.01+20*tf.squeeze(secword,[1]))
            #secword=20*tf.squeeze(secword,[1])/(word-meanword+1e-5)
        else:
            luc,secword=secword
            #secword=tf.squeeze(secword,[1])/(word-meanword)

        ws=tf.square(words-word)
        luc=tf.reduce_mean(tf.reduce_min(ws,axis=-1))
        ws=tf.exp(-ws/(secword+1e-5))/tf.reduce_sum(tf.exp(-ws/(secword+1e-5)),axis=1,keepdims=True)
        #ws=tf.exp(-ws/0.01)/tf.reduce_sum(tf.exp(-ws/0.01),axis=1,keepdims=True)
        #ws=tf.exp(-ws*secword)/tf.reduce_sum(tf.exp(-ws*secword),axis=1,keepdims=True)
        word=tf.reduce_sum(words*ws,axis=1)
        return word,[luc,secword]
    else:
        word=tf.reduce_mean(words,axis=1)
    maxword=tf.reduce_max(words,axis=1)
    #words=tf.squeeze(conv2d('ada_sym',tf.expand_dims(words,axis=2),1,[1,1],stride=[1,1],padding='SAME',stddev=1e-3,activation_func=tf.nn.relu),[2])
    #meanword,varword=tf.nn.moments(words,[1])
    #word=fully_connect('word_out1',word,256,activation_func=tf.nn.relu)
    #word=fully_connect('word_out',word,512,activation_func=tf.nn.relu)
    #word=decoder_with_fc_only(word, layer_sizes=[256, 512], b_norm=True, scope='decode_fc')
    if normal:
        word=word/(tf.reduce_sum(word,axis=-1,keepdims=True)+1e-5)
    return word,maxword
#def local_loss_net(input_signal,n_filter=[64,128,256],activation_func=tf.nn.sigmoid,normal=False):
#    with tf.variable_sc pe('local_loss'):
#        encoder = encoder_with_convs_and_symmetry
#        enc_args = {'n_filters': n_filter,
#                        'filter_sizes': [1],
#                        'strides': [1],
#                        'b_norm': True,
#                        'verbose': True
#                        }
#        #word=encoder(input_signal,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],symmetry=tf.reduce_max,strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
#        words=encoder(input_signal,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],symmetry=None,non_linearity=activation_func,strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
#        word=tf.reduce_max(words,axis=1)
#        if normal:
#            word=word/(tf.reduce_sum(word,axis=-1,keepdims=True)+1e-5)
#    return word,words
#def local_loss_net(input_signal,n_filter=[64,128,256],activation_func=tf.nn.sigmoid,normal=False):
#    with tf.variable_scope('local_loss'):
#        #words=tf.expand_dims(input_signal,axis=2)
#        words=input_signal
#        for i,outchannel in enumerate(n_filter[:-1]):
#            words=conv2d('basic_state1%d'%i,words,outchannel,[1,1],padding='VALID')
#        words=conv2d('basic_stateout',words,n_filter[-1],[1,1],padding='VALID',activation_func=activation_func) 
#        word=tf.reduce_max(words,axis=1)
#        if normal:
#            word=word/(tf.reduce_sum(word,axis=-1,keepdims=True)+1e-5)
#    return word,words
#outword:batch*2048*128
def interpolate_net(pts,cenword,input_feat,n_filter=[64,64],use_net=True,activation_func=tf.nn.relu,reuse=False):
    with tf.variable_scope('dis_interpolation',reuse=reuse):
        decay_ratio=0.1
        cenword=tf.expand_dims(cenword,axis=1)
        dimnum=cenword.get_shape()[-1].value
        dismat=cenword-input_feat
        #tensor=tf.concat([tf.tile(cenword,[1,tf.shape(input_feat)[1],1]),dismat],axis=-1)
        tensor=dismat
        tensor=tf.expand_dims(tensor,axis=2)
        for i,outchannel in enumerate(n_filter[:-1]):
            tensor=conv2d('inter_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=None)
            #tensor=batch_normalization(tensor,name='inter_norm%d'%i)
            tensor=tf.nn.relu(tensor)
        tensor=conv2d('inter_layerout',tensor,3,[1,1],padding='VALID',activation_func=None)
        #tensor=batch_normalization(tensor,name='inter_normout')
        #tensor=activation_func(tensor)
        tensor=tf.squeeze(tensor,axis=[2])
        outword=tensor-pts
    return outword
def reverse_net(scope,raw_feat,input_feat,n_filter=[128,128],activation_func=None):
    with tf.variable_scope(scope):
        raw_feat=tf.expand_dims(raw_feat,axis=2)
        input_feat=tf.expand_dims(input_feat,axis=2)
        raw_length=raw_feat.get_shape()[-1].value
        words=input_feat
        for i,outchannel in enumerate(n_filter):
            words=conv2d('basic_state1%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            #words = batch_normalization(words,name= 'basic_norm%d'%i)
            words=tf.nn.relu(words)
        words=conv2d('basic_stateout',words,raw_length,[1,1],padding='VALID',activation_func=None)
        #words = batch_normalization(words,name= 'basic_normout')
        if activation_func is not None:
            words=activation_func(words)
        diff_words=raw_feat-words
    return diff_words
def rbf_oper(data,cenpts,ratio=0.1):
    data1=tf.expand_dims(data,axis=2)
    dis=tf.reduce_sum(tf.square(data1-cenpts),axis=-1)#batch*2048*n
    maxdis=tf.reduce_sum(tf.square(tf.reduce_max(data,axis=1)-tf.reduce_min(data,axis=1)))
    dis=tf.exp(-ratio*dis/(maxdis))
    #dis=tf.concat([data,dis],axis=-1)
    #mask=tf.exp(-decay_factor*dis)
    #newdata=data*tf.expand_dims(mask,axis=-1)#batch*2048*n*3
    return dis
#cens:batch*n*feat_length
#paras:batch*n*2feat_length
#input_feat:batch*2048*feat_length
#ratios:batch*2048*n*1
def cal_ratio(cens,paras,input_feat,order_num,move,only_cen='c',use_gradient=True,knum=32,re_time=2):
    cens=tf.expand_dims(cens,axis=1)
    input_feat=tf.expand_dims(input_feat,axis=2)
    if paras is not None:
        paras=tf.expand_dims(paras,axis=1)
    move=tf.expand_dims(move,axis=1)
    ptnum=input_feat.get_shape()[1].value
    cennum=cens.get_shape()[2].value
    rnum=move.get_shape()[-1].value
    #print(cennum,ptnum,move)
    if use_gradient:
        base=tf.ones(shape=[tf.shape(paras)[0],ptnum,cennum,1])
    else:
        base=tf.ones(shape=[tf.shape(paras)[0],knum,cennum,1])
    maxwords=None
    if only_cen !='c':
        for i in range(order_num):
            base=tf.concat([base,tf.pow(input_feat-cens,i+1)],axis=-1)#batch*2048*n*2feat_length

    if only_cen=='c':
        #rawratio=tf.reduce_sum(tf.square(input_feat-cens),axis=-1,keepdims=True)#batch*2048*n*1
        
        rawratio=tf.square(input_feat-cens)#batch*2048*n*feat_length
        xyzratio=tf.exp(-10*rawratio)+move##batch*2048*n*feat_length
        xyzratio=tf.minimum(xyzratio,1.0)
        ratio=tf.reduce_prod(xyzratio,axis=-1,keepdims=True)#batch*2048*n*1
        if use_gradient:
            new_feat=input_feat
        else:
            ratio_trans=tf.transpose(ratio,[0,3,2,1])
            _,kid=tf.nn.top_k(ratio_trans,knum)#batch*1*n*knum
            kid=tf.transpose(kid,[0,3,2,1])#batch*knum*n*1
            bnum=tf.shape(input_feat)[0]
            bid=tf.tile(tf.reshape(tf.range(bnum,dtype=tf.int32),[-1,1,1,1]),[1,knum,cennum,1])
            nid=tf.tile(tf.reshape(tf.range(cennum,dtype=tf.int32),[1,1,-1,1]),[bnum,knum,1,1])
            idx=tf.concat([bid,kid,nid],axis=-1)
            new_feat=tf.gather_nd(input_feat,idx)#batch*knum*n*3

        for i in range(order_num):
            base=tf.concat([base,tf.pow(new_feat-cens,i+1)],axis=-1)#batch*2048*n*2feat_length 
        baseratio=tf.square(tf.reduce_sum(base*paras,axis=-1,keepdims=True)-1)
        baseratio=tf.exp(-10*baseratio)#batch*2048/knum*n*1
        #ratio=ratio*baseratio
        if not use_gradient:
            filtered_ratio=tf.where(tf.greater(baseratio,0.9),tf.ones_like(baseratio),tf.zeros_like(baseratio))
            maxwords=filtered_ratio*new_feat
        else:
            #ratio=ratio*baseratio
            maxwords=ratio*new_feat
            
        #rawratio=(rawratio-tf.reduce_min(rawratio,axis=1,keepdims=True))/(1e-5+tf.reduce_max(rawratio,axis=1,keepdims=True)-tf.reduce_min(rawratio,axis=1,keepdims=True))
        #move=tf.expand_dims(move,axis=1)
        #rawratio=tf.expand_dims(rawratio,axis=-1)
        #ratio=tf.exp(-move*rawratio)
        #ratio=tf.exp(-tf.maximum(move,10)*rawratio)
        #ratio=tf.minimum(move+tf.exp(-100*rawratio),1.0)
        #ratio=tf.exp(-10*rawratio)+0.2
        #ratio=tf.reshape(ratio,[-1,ptnum,cennum,1])
    elif only_cen=='m':
        rawratio=tf.square(tf.reduce_sum(base*paras,axis=-1,keepdims=True))#batch*2048*n*1
        #ratio=tf.minimum(tf.nn.relu(rawratio),1.0)
        #ratio=tf.minimum(rawratio,1.0)
        ratio0=rawratio
        new_cennum=cennum
        old_cennum=1
        for r in range(re_time):
            move=tf.tile(move,[1,1,1,old_cennum])
            ratio=(-tf.reduce_min(ratio0,axis=1,keepdims=True)+ratio0)/(tf.reduce_max(ratio0,axis=1,keepdims=True)-tf.reduce_min(ratio0,axis=1,keepdims=True)+1e-5)
            #ratio=tf.minimum(ratio+tf.minimum(tf.reshape(move,[-1,1,new_cennum,1]),0.1),1.0)
            ratio=tf.nn.relu(ratio-0.001)
            #ratio=tf.exp(-100*rawratio)*tf.exp(-10*tf.square(1-tf.minimum(tf.reduce_max(rawratio,axis=1,keepdims=True),1.0)))
            ratio=tf.exp(-10*ratio)
            if r<1:
                rawratio=ratio
            if r==re_time-1:
                #print('ratio',rawratio,ratio)
                #ratio=tf.concat([rawratio,ratio],axis=2)
                break
            old_cennum=new_cennum
            new_cennum=int(new_cennum*cennum)
            ratio=tf.reshape(ratio,[-1,ptnum,1,old_cennum])
            #ratio=tf.tile(ratio,[1,1,cennum,1])
            ratio0=tf.reshape(ratio0*ratio,[-1,ptnum,new_cennum,1])
            ratio=tf.reshape(tf.tile(ratio,[1,1,cennum,1]),[-1,ptnum,new_cennum,1])
            
            maxratio0=tf.reduce_max(ratio0,axis=1,keepdims=True)#batch*1*new_cennum*1
            ratio0=(tf.ones(shape=tf.shape(ratio))-ratio)*maxratio0+ratio0
            #ratio=10*tf.nn.relu(ratio-0.9)
    else:
        #downlimit=tf.reduce_min(input_feat,axis=1,keepdims=True)+paras-1
        #uplimit=tf.reduce_max(input_feat,axis=1,keepdims=True)+cens+1
        downlimit=paras-1.0
        uplimit=paras+cens+1.0
        feat1=tf.nn.relu(uplimit-input_feat)
        feat2=tf.nn.relu(input_feat-downlimit)
        #feat=tf.nn.relu(uplimit-downlimit-tf.nn.relu(input_feat-downlimit))
        
        #feat=tf.reduce_min(feat1,axis=-1,keepdims=True)*tf.reduce_min(feat2,axis=-1,keepdims=True)
        feat=tf.reduce_prod(tf.minimum(feat1,feat2),axis=-1,keepdims=True)
        
        rawratio=(-tf.reduce_min(feat,axis=1,keepdims=True)+feat)/(tf.reduce_max(feat,axis=1,keepdims=True)-tf.reduce_min(feat,axis=1,keepdims=True)+1e-5)
        #rawratio=feat/tf.maximum(feat,1e-5)
        #rawratio=tf.minimum(1.0,1e3*tf.nn.relu(feat-1e-3))
        ratio=tf.minimum(rawratio+0.2,1.0)
        #rawratio=tf.exp(100*tf.reduce_min(feat,axis=-1,keepdims=True))-1.0
        #ratio=tf.minimum(rawratio,1.0)
        
        #rawratio=tf.exp(-tf.square(input_feat-cens)*tf.maximum(move,0.001))
        #ratio=tf.reduce_prod(rawratio,axis=1,keepdims=True)
    #else:
    #    rawratio=tf.reduce_prod
    #ratio=tf.exp(-100*ratio)*tf.exp(-tf.maximum(move,0.1)*disratio)
    #ratio=tf.exp(-tf.maximum(move,0.1)*disratio)
    return maxwords,rawratio,ratio
def filter_by_list(ratio,rawfeat,rlist=[0,0.4,0.8,1]):
    length=len(rlist)
    for i in range(length-1):
        smallratio=tf.nn.relu(ratio-rlist[i])
def find_diff(inwords,mlp=[64,64]):
    words=tf.expand_dims(inwords,axis=2)
    for i,outchannel in enumerate(mlp):
        words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
        #words = batch_normalization(words,name= 'find_diff_norm%d'%i)
        #words=tf.nn.leaky_relu(words,alpha=0.01)
        words=tf.nn.relu(words)
    result=tf.reduce_max(words,axis=1)
    return tf.squeeze(result,axis=1)
def clossnet(data,mlp=[128,128],outlen=128):
    words=tf.expand_dims(data,axis=2)
    zwords=tf.zeros([tf.shape(words)[0],tf.shape(words)[1],tf.shape(words)[2],outlen-3])
    words=tf.concat([words,zwords],axis=-1)
    words0=words[:,:,:,:outlen//2]
    words1=words[:,:,:,outlen//2:]

    mat=words0
    for i,outchannel in enumerate(mlp):
        mat=conv2d('basic_state%d'%i,mat,outchannel,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)
    mat=conv2d('basic_out0',mat,outlen//2,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)
    #words=tf.concat([words[:,:,:,:outlen//2],words[:,:,:,outlen//2:]+words0],axis=-1)
    words1=words1+mat


    #words0=words[:,:,:,outlen//2:]
    mat=words1
    for i,outchannel in enumerate(mlp):
        mat=conv2d('basic_state1%d'%i,mat,outchannel,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)
    mat=conv2d('basic_out1',mat,outlen//2,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)
    #words=tf.concat([words[:,:,:,:outlen//2],words[:,:,:,outlen//2:]+words0],axis=-1)
    words0=words0+mat
    words=tf.concat([words0,words1],axis=-1)
    #print(infeat)
    #infeat=tf.reshape(infeat,[-1,1,1,infeat.get_shape()[-1].value])
    #words=tf.concat([words,tf.tile(infeat,[1,tf.shape(data)[1],1,1])],axis=-1)
    #for i,outchannel in enumerate(mlp):
    #    words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
    #    #words = batch_normalization(words,name= 'find_diff_norm%d'%i)
    #    #words=tf.nn.leaky_relu(words,alpha=0.01)
    #    words=tf.nn.relu(words)

    result=tf.reduce_mean(words,axis=[1,2])
    return result
#input_feat:batch*64*(3+featlen)
#out:batch*64*64*3
def get_auchor_point(scope,input_signal,input_feat,mlp=[64,128],mlp2=[256,256],out_num=64,out_len=3,startcen=None,out_activation=None):
    with tf.variable_scope(scope):
        ftnum=input_feat.get_shape()[1].value
        ptnum=input_signal.get_shape()[1].value
        words=tf.expand_dims(input_feat,axis=2)
        #if startcen is not None:
        #    words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1])],axis=-1)

        #_,input_signal=sampling(ptnum,input_signal,use_type='r')
        knum=2048
        #samnum=int(out_num//(ptnum/knum))
        #input_signal=tf.reshape(input_signal,[-1,knum,3])
        _,startcen=sampling(int(out_num),input_signal,use_type='r')
        #startcen0=tf.reshape(startcen,[-1,out_num,3])
        #words=tf.concat([tf.expand_dims(startcen,axis=2),\
        #        tf.reshape(tf.tile(words,[1,tf.shape(startcen)[1],ptnum//knum,1]),[-1,tf.shape(startcen)[1],1,words.get_shape()[-1].value])],axis=-1)
        words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1])],axis=-1)
        #words=tf.reshape(words,[-1,samnum,1,words.get_shape()[-1].value])
        #words=tf.expand_dims(startcen,axis=2)
        for i,outchannel in enumerate(mlp2):
            words=conv2d('start_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            words=tf.nn.relu(words)
        #wordsfeat=words
        words=tf.reduce_mean(words,axis=1,keepdims=True)
        #print('***********')
        #words=tf.concat([words,tf.expand_dims(input_feat,axis=2)],axis=-1)
        words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp):
            words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            #words = batch_normalization(words,name= 'basic_norm%d'%i)
            #words=tf.nn.leaky_relu(words,alpha=0.01)
            words=tf.nn.relu(words)
        words=conv2d('basic_stateoutg',words,out_len,[1,1],padding='VALID',activation_func=out_activation)
        #print('**************',words)
        move=tf.squeeze(words,axis=2)[:,:,:3]
        newcen=tf.expand_dims(move+startcen,axis=1)
        #words=tf.reshape(words,[-1,ftnum,out_num,out_len])
        #print(startcen,words)
        #newcen=tf.expand_dims(startcen,axis=1)+words[:,:,:,:3]
        words=tf.concat([newcen,tf.reshape(words[:,:,:,3:],[-1,1,newcen.get_shape()[2].value,1])],axis=-1)
        #print(words)
        #words=tf.reshape(words,[-1,ftnum,out_num,out_len])
        #words=conv2d('basic_stateoutg',words,out_len,[1,1],padding='VALID',activation_func=out_activation)
        #words=tf.reshape(words,[-1,out_num,out_len])
        #if startcen is not None:
        #    words=tf.concat([startcen+words[:,:,:3],words[:,:,3:]],axis=-1)
    return words#,move
def get_auchor_fc(scope,input_signal,input_feat,mlp=[64,128],mlp2=[256,256],out_num=64,out_len=3,startcen=None,out_activation=None):
    with tf.variable_scope(scope):
        ptnum=input_feat.get_shape()[1].value
        words=tf.expand_dims(input_feat,axis=2)
        for i,outchannel in enumerate(mlp):
            words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            #words = batch_normalization(words,name= 'basic_norm%d'%i)
            #words=tf.nn.leaky_relu(words,alpha=0.01)
            words=tf.nn.relu(words)
        words=conv2d('basic_stateoutg',words,out_num*out_len,[1,1],padding='VALID',activation_func=out_activation)
        words=tf.reshape(words,[-1,ptnum,out_num,out_len])
    return words#,words
def get_topk(rawcode,codepool,knum):
    valdist,ptid = tf_grouping.knn_point(knum, codepool, rawcode)#batch*n*k
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(rawcode)[0],dtype=tf.int32),[-1,1,1,1]),[1,tf.shape(rawcode)[1],knum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    kcode=tf.gather_nd(codepool,idx)#batch*n*k*c
    #kdist=tf.reduce_mean(tf.square(tf.expand_dims(rawcode,axis=2)-kcode),axis=-1)
    return idx,kcode
def fc_layers(words,dim,mlp=[128,128]):
    #print(words)
    #assert False
    words=tf.reshape(words,[-1,1,1,words.get_shape()[-1].value])
    for i,outchannel in enumerate(mlp):
        words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
    words=conv2d('basic_stateoutg',words,dim*4,[1,1],padding='VALID',activation_func=None)
    result=tf.squeeze(words,[1,2])
    #print(words)
    return result
#input_pts:batch*2048*1*3
#auchor_pts:batch*1*64*3
def auchor_feat_point(input_feat,input_pts,auchor_pts,directions=None,use_gradient=True,use_acti=False,use_direction=True):
    auchor_pts,auchor_rs=auchor_pts[:,:,:,:3],auchor_pts[:,:,:,3:]
    #print('auchor',auchor_rs)
    #knum=64
    #in_renum=input_pts.get_shape()[2].value
    #au_renum=auchor_pts.get_shape()[2].value
    #vecs=tf.expand_dims(input_pts,axis=3)-tf.expand_dims(auchor_pts,axis=1)#batch*2048*64*64*3,batch*k*region_num*auchor_num*3
    vecs=input_pts-auchor_pts
    #dist=tf.sqrt(tf.squeeze(tf.matmul(tf.expand_dims(vecs,axis=3),tf.expand_dims(vecs,axis=-1)),[-1]))
    #print(dist)

    #dist=tf.sqrt(tf.reduce_sum(tf.square(vecs),axis=-1,keepdims=True))
    
    #dist=tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(tf.tile(input_pts,[1,1,1,1])),axis=-1,keepdims=True)+\
    #        tf.reduce_sum(tf.square(tf.tile(auchor_pts,[1,1,1,1])),axis=-1,keepdims=True)-\
    #        2*tf.reduce_sum(input_pts*auchor_pts,axis=-1,keepdims=True)))
    #dist=dist1+tf.stop_gradient(dist-dist1)
    #print(input_pts,auchor_pts,dist,input_pts*auchor_pts)
    #assert False
    ptnum=input_pts.get_shape()[1].value
    anum=auchor_pts.get_shape()[2].value
    knum=2048
    fnum=ptnum//knum
    #inpts=tf.reshape(input_pts,[-1,knum,1,3])#4b*512*1*3
    #apts=tf.reshape(auchor_pts,[-1,1,anum//fnum,3])#4b*1*32*3
    #ars=tf.reshape(auchor_rs,[-1,1,anum//fnum,1])#4b*1*32*3
    #apts=tf.tile(auchor_pts,[fnum,1,1,1])
    #ars=tf.tile(auchor_rs,[fnum,1,1,1])
    #inpts=tf.reshape(input_pts,[-1,knum,fnum,3])
    #inpts=tf.tile(inpts,[1,1,auchor_pts.get_shape()[-2].value//fnum,1])

    inpts=input_pts
    apts=auchor_pts
    ars=auchor_rs

    dist=tf.sqrt(1e-4+tf.reduce_sum(tf.square(inpts)+\
             tf.square(apts)-2*inpts*apts,axis=-1,keepdims=True))-0.01
    #print(input_pts,auchor_pts,dist)
    #assert False
           #2*tf.transpose(tf.matmul(tf.transpose(input_pts,[0,2,1,3]),tf.transpose(auchor_pts,[0,1,3,2])),[0,2,3,1])
    #dist=tf.sqrt(1e-8+dist)
    #print('...............',dist)
    #_,vecs=get_topk(tf.squeeze(auchor_pts,[1]),tf.squeeze(input_pts,[2]),knum=knum)#batch*128*32*3
    #vecs=vecs-tf.reshape(auchor_pts,[-1,au_renum,1,3])
    #vecs=tf.reshape(vecs,[-1,knum,1,au_renum,3])
    #dist=tf.sqrt(tf.reduce_sum(tf.square(vecs),axis=-1))
    #rawratio=dist
    bsize=16
    featvec=None
    #if use_gradient:
    rawratio=tf.exp(-dist/(0.01+tf.square(ars)))
    #ratio=tf.squeeze(rawratio,[2])
    #print(input_pts)
    #assert False
    rsum=tf.reduce_sum(rawratio,axis=[1],keepdims=True)
    ratio=rawratio/(1e-8+rsum)
    #print('...............',ratio,rawratio,rsum,dist,auchor_rs)
    #print(rawratio,ratio)
    #assert False
    trnum=16#inpts.get_shape()[1].value/apts.get_shape()[2].value
    #trnum=ptnum*ptnum/(inpts.get_shape()[1].value*auchor_pts.get_shape()[2].value)
    #print(trnum)
    #assert False
    #trnum=1.0
    #print(auchor_pts)
    #assert False
    #print(inpts,ratio,dist)
    featvec1=trnum*tf.reduce_sum(inpts*ratio,axis=1)-trnum*(1-1e-8/(1e-8+tf.squeeze(rsum,[1])))*tf.squeeze(apts,[1])
    #print(featvec1)
    featvec1=tf.reshape(featvec1,[-1,1,anum,3])
    #print(featvec1)
    #print(dist)
    #assert False
    #dist=tf.reshape(dist,[-1,knum,anum,1])
    #featvec1=16*tf.reduce_sum(ratio*vecs,axis=1)

    #print(featvec1,dist)
    #assert False
    #featvec1=tf.reduce_sum(vecs*ratio,axis=1)
    #numfeat=tf.reduce_sum(rawratio,axis=1)
    ##numfeat=(numfeat)/(1e-8+tf.reduce_sum(numfeat,axis=2,keepdims=True))
    ##featvec1=tf.concat([featvec1,0.01*numfeat],axis=-1)
    #if featvec is not None:
    #    featvec=tf.concat([featvec,featvec1],axis=-1)
    #else:
    #    featvec=featvec1
    #featvec=tf.concat([featvec,numfeat],axis=-1)
    #featvec=tf.concat([featvec,tf.reduce_max(ratio,axis=1)],axis=-1)
    #else:
    #    bnum=tf.shape(input_pts)[0]
    #    pid=tf.expand_dims(tf.argmin(rawratio,axis=1,output_type=tf.int32),axis=-1)#batch*region_num*auchor_num*1
    #    bid=tf.tile(tf.reshape(tf.range(bnum,dtype=tf.int32),[-1,1,1,1]),[1,in_renum,au_renum,1])
    #    rid=tf.tile(tf.reshape(tf.range(in_renum,dtype=tf.int32),[1,-1,1,1]),[bnum,1,au_renum,1])
    #    aid=tf.tile(tf.reshape(tf.range(au_renum,dtype=tf.int32),[1,1,-1,1]),[bnum,in_renum,1,1])
    #    idx=tf.concat([bid,pid,rid,aid],axis=-1)
    #    featvec=tf.gather_nd(vecs,idx)
        
    #featvec=tf.reshape(featvec1,[bsize,in_renum,-1])
    #if use_acti:
    #    result=tf.exp(-dist)
    #else:
    #    result=dist
    #print(dist,result)
    return dist,featvec1
def folding_pts(scope,input_tensor,grid_size,grid_length=1.0):
    xgrid_size=grid_size[0]
    ygrid_size=grid_size[1]
    up_ratio=xgrid_size*ygrid_size
    xlength,ylength=grid_length 
    xgrid_feat=-xlength+2*xlength*tf.tile(tf.reshape(tf.linspace(0.0,1.0,xgrid_size),[1,1,-1,1]),[tf.shape(grid_feat)[0],tf.shape(grid_feat)[1],1,1])#batch*1*xgrid*1
    ygrid_feat=-ylength+2*ylength*tf.tile(tf.reshape(tf.linspace(0.0,1.0,ygrid_size),[1,1,-1,1]),[tf.shape(grid_feat)[0],tf.shape(grid_feat)[1],1,1])#batch*1*ygrid*1
    grid_feat=tf.concat([tf.tile(xgrid_feat,[1,1,ygrid_size,1]),tf.reshape(tf.tile(ygrid_feat,[1,1,1,xgrid_size]),[-1,ptnum,up_ratio,1])],axis=-1)#batch*1*up_ratio*2
    
    new_state=tf.concat([tf.tile(input_tensor,[1,1,up_ratio,1]),grid_feat],axis=-1) 
def local_loss_net(input_signal,all_feat,rawmask=None,activation_func=tf.nn.sigmoid,only_cen='c',use_gradient=True):
    with tf.variable_scope('local_loss'):
        ptnum=input_signal.get_shape()[1].value
        feat_length=input_signal.get_shape()[-1].value
        gfeat=all_feat
        gfeat=tf.expand_dims(gfeat,axis=1)
        auchor_pts=None
        if rawmask is not None:
            auchor_pts=rawmask

        if auchor_pts is None:
            gauchor_pts=get_auchor_point('auchor_layerg',input_signal,gfeat,mlp=[128,64],out_num=128,out_len=4,startcen=None,out_activation=None)#88
            #gauchor_pts=get_auchor_fc('auchor_layerg',input_signal,gfeat,mlp=[256,256,256],out_num=32,out_len=4,startcen=None,out_activation=None)
            auchor_pts=gauchor_pts

        #words,rawratio,ratio=cal_ratio(cens,paras,input_signal,order_num,move,only_cen=only_cen,re_time=2,use_gradient=use_gradient)#batch*2048*n*1
        #regions=region_num*region_num
        #gregion=int(0.5*regions)
        #gratio=ratio 
        #coor_words=ratio*tf.expand_dims(input_signal,axis=2)
        dist,feat_vec=auchor_feat_point(gfeat,tf.expand_dims(input_signal,axis=2),auchor_pts,use_acti=False,use_gradient=use_gradient,use_direction=False)
        word=feat_vec
        print(dist,auchor_pts)
        #assert False
        reverse_word=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(dist[:,:,:112,:],axis=-2),axis=1),axis=1,keepdims=True)\
                +0.1*tf.reduce_mean(tf.reduce_max(tf.square(auchor_pts[:,:,:,3:]),axis=-1),axis=-1)#56
        lr=0.1*tf.reduce_mean(tf.reduce_max(tf.square(auchor_pts[:,:,:,3:]),axis=-1),axis=-1)
        rawmask=auchor_pts
    return word,reverse_word,rawmask,lr
def local_loss_net2(input_signal,input_feat,all_feat,inmask=None,n_filter=[64,128,256],activation_func=tf.nn.sigmoid,normal=False,use_mask=False):
    with tf.variable_scope('local_loss'):
        ptnum=input_signal.get_shape()[1].value
        all_feat=tf.expand_dims(all_feat,axis=1)
        #words=tf.expand_dims(tf.concat([input_signal,input_feat,tf.tile(all_feat,[1,tf.shape(input_signal)[1],1])],axis=-1),axis=2)
        #words=tf.expand_dims(tf.concat([input_signal,input_feat],axis=-1),axis=2)
        #words=tf.expand_dims(input_signal,axis=2)
        words=tf.expand_dims(all_feat,axis=1)
        if use_mask and inmask is not None:
            mask=inmask
            meanmask,varmask=tf.nn.moments(mask,[1])
            reverse_mask=1-mask
            if normal:
                reverse_mask=tf.exp(reverse_mask)/(tf.reduce_sum(tf.exp(reverse_mask),axis=-1,keepdims=True)+1e-5)
        else:
            for i,outchannel in enumerate(n_filter[:-1]):
                words=conv2d('basic_state1%d'%i,words,outchannel,[1,1],padding='VALID')
                
            words=conv2d('basic_stateout',words,n_filter[-1],[1,1],padding='VALID',activation_func=activation_func)
            rawmask=tf.squeeze(words,axis=[2])
            meanmask,varmask=tf.nn.moments(rawmask,[1])
            if normal:
                mask=tf.exp(rawmask)/(tf.reduce_sum(tf.exp(rawmask),axis=-1,keepdims=True)+1e-5)
                reverse_mask=tf.reduce_max(rawmask,axis=-1,keepdims=True)-rawmask
                reverse_mask=tf.exp(reverse_mask)/(tf.reduce_sum(tf.exp(reverse_mask),axis=-1,keepdims=True)+1e-5)
            else:
                mask=rawmask
                reverse_mask=1-mask
        words=mask*input_feat
        #reverse_mask=tf.reduce_max(rawmask,axis=-1,keepdims=True)-rawmask
        #reverse_mask=1-mask
        #if normal:
        #    reverse_mask=tf.exp(reverse_mask)/(tf.reduce_sum(tf.exp(reverse_mask),axis=-1,keepdims=True)+1e-5)
        #balance_ratio=tf.reduce_max(mask,axis=1,keepdims=True)/tf.reduce_max(reverse_mask,axis=1,keepdims=True)
        reverse_words=reverse_mask*input_feat
        word=tf.reduce_max(words,axis=1)
        reverse_word=tf.reduce_max(reverse_words,axis=1)
    return word,words,reverse_word,tf.reduce_mean(tf.sqrt(varmask)),mask
def cen_net(data,mlp1=[64,64],mlp2=[64],cen_num=8):
    state=tf.expand_dims(data,axis=2)
    for i,outchannel in enumerate(mlp1):
        state=conv2d('basic_state1%d'%i,state,outchannel,[1,1],padding='VALID')
    state=tf.reduce_max(state,axis=1,keepdims=True)
    for i,outchannel in enumerate(mlp2):
        state=conv2d('basic_state2%d'%i,state,outchannel,[1,1],padding='VALID')
    state=conv2d('cenpts%d'%i,state,cen_num*3,[1,1],padding='VALID',activation_func=None)#batch*1*1*4n
    cenpts=tf.reshape(state,[-1,1,cen_num,3])
    #decay_ratio=tf.nn.relu(cenpts[:,:,:,-1])
    return cenpts,tf.constant(0.1)

def discriminator(word,n_filter=[128,64,64],bnorm=False,reuse=False):
    net=word
    for i in range(len(n_filter)):
        net=fully_connect('dis_fulcon%d'%i,net,n_filter[i],activation_func=tf.nn.relu)
        if bnorm:
           net=batch_normalization(net,) 
    poss=fully_connect('poss_out',net,1,activation_func=tf.nn.sigmoid)
    return poss



def default_train_params(single_class=True):
    params = {'batch_size': 50,
              'training_epochs': 500,
              'denoising': False,
              'learning_rate': 0.0005,
              'z_rotate': False,
              'saver_step': 10,
              'loss_display_step': 1
              }

    if not single_class:
        params['z_rotate'] = True
        params['training_epochs'] = 1000

    return param
