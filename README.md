# Segmentation of OCT Images with DME using UNet

![alt text](https://github.com/surengunturumasters/deeplearning_final/blob/main/eye.png)

## Introduction
Problem: A deep learning approach to diagnose diabetic macular edema using Optical Coherence Tomography

DME (Diabetic Macular Edema) is caused by type-II diabetes and it usually affects the eye. This can ultimately result in blindness. In fact, 750,000 Americans are diagnosed with this particular disease so it is of utmost importance to be able to identify DME before it becomes serious. 

One particular test for DME is Optical Coherence Tomography (OCT), which uses a light source to show the inner cell layers of the retina. This can be used to determine the amount of swelling of the retina. 

## Dataset
The main dataset used in this project was the segmentation of OCT images from the Optical Coherence Tomography and Diabetic Macular Edema dataset on kaggle by Paul Mooney. The link to the dataset is shown [here](https://www.kaggle.com/paultimothymooney/chiu-2015). The training set includes information from six patients while the validation set includes information from one patient. The target classes for classification was found from segmented fluid-filled regions and segmented retinal layer boundaries. More information about the dataset can be found using the link. 

## Methods
The deep learning approach that we tried to implement for this particular use case is a Unet. Usually, traditional CNN's would be used for image classification, but this is an image segmentation task where a label needs to be shown for every pixel. Because of this, UNets had to be used so that it would create a label around a neighborhood of pixels for a certain pixel, and then upsample those results so that the label would be shown in that particular pixel. Because of this, the output of the UNet had to be the size of the target image. The UNet architecture is shown below: 

![alt text](https://github.com/surengunturumasters/deeplearning_final/blob/main/unet_arch.png)

However with this architecture, we varied the amount of encoding and decoding blocks in our implementation to test out what UNet models work best

Our results contain three models: 
1) UNet with four encoding blocks and four decoding blocks as shown [here](https://github.com/surengunturumasters/deeplearning_final/blob/main/unet-model1.ipynb)
2) UNet with three encoding blocks and three decoding blocks as shown [here](https://github.com/surengunturumasters/deeplearning_final/blob/main/unet.ipynb)
3) UNet with two encoding blocks and two decoding blocks as shown [here](https://github.com/surengunturumasters/deeplearning_final/blob/main/Unet_model2.ipynb)

From our results, the UNet with three encoding blocks and three decoding blocks had the best IOU under 100 epochs with a value of 0.22. Although the UNet with four encoding blocks was very close in IOU, it took about an hour to train 100 epochs under a gpu because the final layer of the model contains a convolutional layer with a kernel size of 81 to make sure the dimensions match the output. However, the UNet with three blocks was perfectly able to match the dimensions of the output without exploding the kernel size, and so training time for 100 epochs took only 10 minutes. Therefore, we used model (2) from above as the model for final segmentation. 

## Result
Go to the bottom of the below link to get the images predicted from the final UNet model

[Link of the final UNET notebook images](https://github.com/surengunturumasters/deeplearning_final/blob/main/unet.ipynb)

Trained the UNet model with three encoding blocks on 1000 epochs and got the images as shown in the bottom of the link of the notebook above. From the results, validation loss tends to oscillate a lot, with the best loss coming around epoch 800. However, more data would be needed in order to create a more accurate model, and this was the main difficulty throughout the project. As of now, we saw that the model overfitted on the training set on some epochs and on others, it was alright, which was a bit mysterious. 

The resulting images from the model trained on 1000 epochs is shown below: 

![alt text](https://github.com/surengunturumasters/deeplearning_final/blob/main/results.png)

The comparisons are mainly done between the segmented localization images and the target images and as we can see, some parts of the shapes are being identified but still, not enough to provide accurate results. More data or a more accurate model would need to be produced to work on this further. 

## Next Steps
Other ways to analyze our UNet model is to play around with the patch size or the number of pixels in a neighborhood to predict the class of a certain pixel in an image. We could also play around with using more layers, but the biggest impediment was the lack of data. With more data, the UNet model should be able to learn the associations between the input image and the target segmented image. 

Other ideas for next steps include looking more deeply into different metrics such as accuracy, IOU, dice coefficient, and many more. In our examples, we looked at IOU, but the value kept fluctuating throughout the epochs, where the IOU value after 100 epochs of training for the validation set would be higher than it was after 500 epochs of training, but again would be high again after 600 epochs. Exploring why there are fluctuations in IOU throughout epochs of training would be something to look into for next steps. 

