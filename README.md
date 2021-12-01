# EVSS
Distortion Map-Guided Feature Rectification for Efficient Video Semantic Segmentation

We present an efficient distortion map-guided feature rectification method for video semantic segmentation, specifically targeting the feature updating and correction on the distorted regions with unreliable optical flow. The updated features for the distorted regions are extracted from a light correction network (CoNet). A distortion map serves as the weighted attention to guide the feature rectification by aggregating the warped features and the updated features. The generation of the distortion map is simple yet effective in predicting the distorted areas in the warped features, i.e., moving boundaries, thin objects, and occlusions. 

- Model architecture
<p align="center"><img src="./figure/figure4.png" width="860" alt="" /></p>

- Coarse-to-fine distortion map
<p align="center"><img src="./figure/figure6.png" width="500" alt="" /></p>



## Dependencies

- Python 3.5
- Pytorch 1.5.0
- CUDA 10.2 
- cuDNN 7.6.5

The experiments are conducted on a Ubuntu 18.04 LTS PC with two NVIDIA GeForce GTX 1080 Ti. Driver version is 460.91.03. GCC version is 7.5.0. Please refer to [`requirements.txt`](requirements.txt) for more details.

## Dataset setup

Please download the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) (324G) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) (17G) **video sequences** dataset. For Cityscapes dataset, you need to send an email to Cityscapes requesting permission for the video dataset.

Your directory tree should be look like this:
````bash
$EVSS_root/dataset
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
├── camvid
│   ├── 11labels
│   │   ├── segmentation annotations
│   └── video_image
│       ├── 0001TP
│           ├── decoded images from video clips
│       ├── 0006R0
│       └── 0016E5
│       └── Seq05VD
````

## Training
- Create a virtual environment.
- Install resampled 2d modules for the feature warping process.
```
cd $EVSS_ROOT/model/resample2d_package
python setup.py build
```
- Download the pretrained CSRNet and FlowNet models. 

   Put the pretrained models under $EVSS_ROOT/saved_model/pretrained.

| model | Link |
| :--: | :--: |
| CSRNet_Cityscapes | [Google Drive](https://drive.google.com/file/d/1onVZChvwK25OUW4Now6vgGXSzlnhmK2q/view?usp=sharing) |
| CSRNet_CamVid | [Google Drive](https://drive.google.com/file/d/16e7T4fMarJKIzn5_-e27UhRPmaZgdvzN/view?usp=sharing) |
| FlowNet | [Google Drive](https://drive.google.com/file/d/1xJjhkjVGjKJyPVBfhlELl1Gzse4RFm-_/view?usp=sharing) |

- Run bash file to strat training.
````bash
# train on Cityscapes
bash run/train_citys.sh
# train on CamVid
bash run/train_camvid.sh
````

## Testing
- Run bash file to strat testing.
````bash
# test on Cityscapes
bash run/val_citys.sh
# test on CamVid
bash run/val_camvid.sh
````
## Evaluating our pretrained models
Please download the pretrained EVSS model on Cityscapes and CamVid. 

Put the pretrained EVSS models under $EVSS_ROOT/saved_model/evss.

| model | Link |
| :--: | :--: |
| EVSS_Cityscapes | [Google Drive](https://drive.google.com/file/d/10IlbD334GjQB6p5RiNvSfdQoeNUGrKev/view?usp=sharing) |
| EVSS_CamVid | [Google Drive](https://drive.google.com/file/d/1SLB-r2c6OVTJMsJnmYmN8mYouCpdWMwb/view?usp=sharing) |




