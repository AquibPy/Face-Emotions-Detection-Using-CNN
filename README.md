# Face Emotion detection
![Python](https://img.shields.io/badge/python-3.x-orange.svg)
![Type](https://img.shields.io/badge/Deep-Learning-red.svg) ![Type](https://img.shields.io/badge/Type-KERAS-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-yellowgreen.svg)

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks.The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)

## Basic Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.


* Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing) and unzip it inside the `src` folder.

* If you want to train this model, use:  

```bash
python emotion.py --mode train
```

* If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing) and then run:  

* The folder structure is of the form:  
  src:
  * `app.py`(file)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

* This implementation by default detects emotions on all faces in the webcam feed. With a simple 3-layer CNN, the test accuracy reached 74% in 50 epochs.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

