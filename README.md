# CALoss
[CVPR'23] The codes for Learning to Measure the Point Cloud Reconstruction Loss in a Representation Space

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* numpy 1.14.5

We also provide an available conda environment (`lcd.yaml`) in this repo. Please run:

```
conda env create -f caloss.yaml
```

## Dataset
The adopted ShapeNet Part dataset is adopted following [FoldingNet](http://www.merl.com/research/license#FoldingNet), while the ModelNet10 and ModelNet40 datasets follow [PointNet](https://github.com/charlesq34/pointnet.git). Other datasets can also be used. Just revise the path by the (`--filepath`) parameter when training or evaluating the networks.
The files in (`--filepath`) should be organized as

        <filepath>
        ├── <trainfile1>.h5 
        ├── <trainfile2>.h5
        ├── ...
        ├── train_files.txt
        └── test_files.txt

where the contents in (`train_files.txt`) or (`test_files.txt`) should include the directory of training or testing h5 files, such as:

        train_files.txt
        ├── <trainfile1>.h5
        ├── <trainfile2>.h5
        ├── ...

We also provide the processed datasets in [Google Drive](https://drive.google.com/file/d/1sjUk8o-wsZp2PJUej4TsmjnOPvjJegKR/view?usp=sharing). You can just download, unzip them, and just set filepath to one of the dataset paths.

## Usage

1. Preparation

```
cd ./tf_ops
bash compile.sh
```

2. Train

For the reconstruction task,
```
Python3 vv_cae.py
```

Note that the paths of data should be edited through the (`--filepath`) parameter according to your setting. For example, if we use the download dataset (`./objdata/ShapeNet_part`), the training command would be

```
Python3 vv_cae.py --filepath ./objdata/ShapeNet_part
```

3. Test

For the evaluation of reconstruction errors,
```
Python3 vvae_eva.py --savepath ('weights') --filepath ('dataset')
```

The trained weight files should be provided by the (`--savepath`) parameter to evaluate the performances.

Here, we also provide [weights](https://drive.google.com/file/d/19IqJ-LV5zpVrstu2yCn-LcUnt9mLaPtq/view?usp=sharing) of the reconstruction network AE pre-trained on ShapeNet Part dataset. To evaluate its performance, just download and unzip it, then change the savepath to its folder.

