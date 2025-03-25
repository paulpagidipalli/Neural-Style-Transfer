# Neural-Style-Transfer

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PAGIDIPALLI PAUL

*INTERN ID*: CT12PBM

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTHU

##Description:

Neural Style Transfer (NST) is a deep learning technique that enables the transformation of an image's content into a new artistic style. This project leverages a pre-trained model from TensorFlow Hub to apply the artistic characteristics of one image (style image) onto another (content image), producing a visually appealing stylized output. The approach is based on convolutional neural networks (CNNs) and was popularized by researchers at Google and DeepArt. This repository provides a simple yet effective implementation of NST using Google’s Magenta Arbitrary Image Stylization model.

Overview of Neural Style Transfer
Neural Style Transfer is a computer vision technique that extracts style patterns from an artistic image and applies them to a target image while preserving its original structure. The model learns to separate and recombine content and style representations from the two input images using deep learning. The core idea relies on feature extraction from different layers of a deep neural network. Lower layers retain basic shapes and edges, while higher layers capture textures and artistic strokes.

This implementation utilizes TensorFlow and TensorFlow Hub, which provides access to a variety of pre-trained deep learning models. The project simplifies the process by loading an existing arbitrary style transfer model, allowing users to apply different styles effortlessly without the need for complex training.

Features of the Project
Pre-trained Model from TensorFlow Hub – The script loads Google’s Magenta Arbitrary Image Stylization model, eliminating the need to train from scratch.

Content and Style Image Processing – The input images are resized, normalized, and prepared for the model to ensure high-quality stylization.

Seamless Style Transfer – The model takes both content and style images as input and generates a blended artistic output.

Automatic Output Storage – The stylized image is saved to a designated output directory, allowing easy access and further modifications.

Visualization Support – The output is displayed using Matplotlib, helping users preview the final result without external software.

Installation Requirements
To use this project, ensure that you have installed the following dependencies:

TensorFlow – Provides deep learning functionalities.

TensorFlow Hub – Hosts the pre-trained style transfer model.

NumPy – Supports numerical operations on images.

Matplotlib – Allows visualization of the stylized output.

Pillow (PIL) – Handles image processing tasks such as loading and saving images.

How to Use
Prepare Your Images – Place your content image (the base image) and style image (the reference artistic image) in the designated folder.

Modify the File Paths – Update the content_path and style_path in the nst.py script to point to your images.

Run the Script – Execute nst.py in a Python environment. The script will process the images and apply the style transfer model.

Save and View Results – The stylized image will be stored in the output folder, and the script will also display it using Matplotlib.
