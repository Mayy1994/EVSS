# EVSS
Distortion Map-Guided Feature Rectification for Efficient Video Semantic Segmentation

We present an efficient distortion map-guided feature rectification method for video semantic segmentation, specifically targeting the feature updating and correction on the distorted regions with unreliable optical flow. The updated features for the distorted regions are extracted from a light correction network (CoNet). A distortion map serves as the weighted attention to guide the feature rectification by aggregating the warped features and the updated features. The generation of the distortion map is simple yet effective in predicting the distorted areas in the warped features, i.e., moving boundaries, thin objects, and occlusions. 

<img src="./figure/figure-4.png" width="860"/>

## Dependencies

- Python 3.5
- Pytorch
- CUDA 10.2 
- cuDNN 7.6.5

The experiments are conducted on a Ubuntu 18.04 LTS PC with two NVIDIA GeForce GTX 1080 Ti. Please refer to requirements.txt (requirements.txt) for more details.

## Dataset setup

Please download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid//) **video sequences** dataset.

## Training
- Create a virtual env
- To Install resampled 2d modules
```
cd $DAVSS_ROOT/lib/model/resample2d_package
python setup.py build
```
- Download the pretrained CSRNet and FlowNet
- Run bash file to strat training
````bash
# train on Cityscapes
bash train_citys.sh
# train on CamVid
bash train_camvid.sh
````

## Testing
- Run bash file to strat testing
````bash
# test on Cityscapes
bash val_citys.sh
# test on CamVid
bash val_camvid.sh
````
