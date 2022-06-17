## PVANet: Lightweight Deep Neural Networks for Real-time Object Detection
by Sanghoon Hong, Byungseok Roh, Kye-hyeon Kim, Yeongjae Cheon, Minje Park (Intel Imaging and Camera Technology)
Presented in [EMDNN2016](http://allenai.org/plato/emdnn/), a NIPS2016 workshop ([arXiv link](https://arxiv.org/abs/1611.08588))

### Introduction

This repository is a fork from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and demonstrates the performance of PVANet.

You can refer to [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) and [faster-rcnn README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more information.

### Desclaimer

Please note that this repository doesn't contain our in-house codes used in the published article.
- This version of py-faster-rcnn is slower than our in-house runtime code (e.g. image pre-processing code written in Python)
- PVANet was trained by our in-house deep learning library, not by this implementation.
- There might be a tiny difference in VOC2012 test results, because some hidden parameters in py-faster-rcnn may be set differently with ours.

### Citing PVANet

If you want to cite this work in your publication:
```
@article{hong2016pvanet,
  title={{PVANet}: Lightweight Deep Neural Networks for Real-time Object Detection},
  author={Hong, Sanghoon and Roh, Byungseok and Kim, Kye-Hyeon and Cheon, Yeongjae and Park, Minje},
  journal={arXiv preprint arXiv:1611.08588},
  year={2016}
}
```

### Installation
0. Install Dependency
    ```Shell
       sudo pip install Cython
    ``` 
    
1. Clone the Faster R-CNN repository
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/sanghoon/pva-faster-rcnn.git
    ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

3. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html
    # For your Makefile.config:
    #   Uncomment `WITH_PYTHON_LAYER := 1`

    cp Makefile.config.example Makefile.config
    make -j8 && make pycaffe
    ```

4. Download PVANet detection model for VOC2007
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_voc2007.sh
    ```

5. Download PVANet detection model for VOC2012 (published model)
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_voc_best.sh
    ```    
    
6. (Optional) Download all available models (including pre-trained and compressed models)
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_all_models.sh
    ```

7. (Optional) Download ILSVRC2012 (ImageNet) classification model
    ```Shell
    cd $FRCN_ROOT
    ./models/pvanet/download_imagenet_model.sh
    ```

8. (Optional) If the scripts don't work, please download the models from ...

    |  Model | Google Drive |
    | ------ | ---- |
    | PVANet for VOC2007 | [link](https://drive.google.com/file/d/0Bw_6VpHzQoMVRGZOSEctOEVMLXc/view?usp=sharing&resourcekey=0-V1KjyzIboCo40TgqUlsBFQ) |
    | PVANet for VOC2012 | [link](https://drive.google.com/file/d/0Bw_6VpHzQoMVa3M0Zm5zNnEtQUE/view?usp=sharing&resourcekey=0-luUPOg9MSnrw_A9v1XkhPA) |
    | PVANet for VOC2012 (compressed) | [link](https://drive.google.com/file/d/0Bw_6VpHzQoMVZU1BdEJDZG5MVXM/view?usp=sharing&resourcekey=0-CrLfgh4hqTg8d8KMDM7IDw) |
    | PVANet for ILSVRC2012 (ImageNet) | [link](https://drive.google.com/file/d/0Bw_6VpHzQoMVTjctVVhjMXo1X3c/view?usp=sharing&resourcekey=0-sA75fRqcGHKmCXw8Svka4w) |
    | PVANet pre-trained | [link](https://drive.google.com/file/d/0Bw_6VpHzQoMVak5FVFBWU0Uyb3M/view?usp=sharing&resourcekey=0-63zxVAnC2CXNGqnmnTNWsA) |

### How to run the demo

1. Download PVANet detection models

2. Run the demo script
    ```Shell
    cd $FRCN_ROOT
    ./tools/test_pvanet.py
    ```

### How to test on the VOC dataset

1. Download PASCAL VOC 2007 and 2012
-- Follow the instructions in [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

2. PVANet on PASCAL VOC 2007
    ```Shell
    cd $FRCN_ROOT
    ./tools/test_net.py --net models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712.caffemodel --def models/pvanet/pva9.1/faster_rcnn_train_test_21cls.pt --cfg models/pvanet/cfgs/submit_1019.yml --gpu 0
    ```

3. PVANet (compressed)
    ```Shell
    cd $FRCN_ROOT
    ./tools/test_net.py --net models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel --def models/pvanet/pva9.1/faster_rcnn_train_test_ft_rcnn_only_plus_comp.pt --cfg models/pvanet/cfgs/submit_1019.yml --gpu 0
    ```

### Expected results

#### Mean Average Precision on VOC detection tasks

| Model     | VOC2007 mAP (%) | VOC2012 mAP (%) |
| --------- | ------- | ------- |
| PVANet+ (VOC2007) | **84.9** | N/A |
| PVANet+ (VOC2012) | *89.8* | **84.2** |
| PVANet+ (VOC2012 + compressed) | *87.8* | 83.7 | 
- The training set for the VOC2012 model includes the VOC2007 test set. Therefore the accuracies on VOC2007 of the model are not meaningful; They're shown here just for reference

#### Validation error on ILSVRC2012

| Input size | Top-1 error (%) | Top-5 error (%) |
| --- | --- | --- |
| 192x192 | 30.00 | N/A |
| 224x224 | 27.66 | 8.84 |
- We re-trained a 224x224 model from the '192x192' model as a base model.

