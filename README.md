# tfFoodImageClassifier
Food or non-food? Image Classification with TensorFlow ANN

### Download project
```
git clone https://github.com/enyangxxx/tfFoodImageClassifier.git
```

### Download dataset
1. You can download the dataset Food-5K here by using e.g. Cyberduck to access via FTP:
https://mmspg.epfl.ch/downloads/food-image-datasets/

2. Create a folder 'images' in project root folder:
```
cd foodImageClassfier && mkdir images
```

3. Copy the sub-folders 'training', 'evaluation', 'validation' into the 'images' folder. 

### About the dataset
The images with name starting with 1 are food images, names of non-food images start with 0. The training set contains 3000 images (50:50), the evaluation and validation sets contain both 1000 examples with an equal distribution between food images and non-food images.

### Hyperparameters
In this project, I learned to use TensorFlow to build an Deep Neural Network and also how to choose different hyperparameters. The following values are chosen based on given circumstances, e.g. the small size of dataset.

- Number of epochs = 2000
- Size of mini-batches = 1000
- Learning rate = 0.0001
- Side length of an image = 20
- Number of layers = 8
- Dimensions of the layers (or call it number of units per layer) = [1200, 500, 100, 80, 50, 40, 10, 2]

### Training result
The costs after each 100th epoch are the following:

<img src="https://github.com/enyangxxx/tfFoodImageClassifier/blob/master/gitImg/costs.jpg" width="200" height="350">

They were also plotted in a learning graph, together with the training, cross-validation and test accuracies:

<img src="https://github.com/enyangxxx/tfFoodImageClassifier/blob/master/gitImg/learningcurveAndAccuracies.jpg" width="500" height="400">

This reveals an effect of overfitting, because the model fits the training set very well, but less to the cross-validation set due to small size of dataset overall. A possible solution could be data augmentation to enlarge the dataset. 

## Test result
This is my first time ever showing the test result with single test images.. I am very very proud to achieve this result although I know that the model can still be improved. But my goal is definitely achieved and a big thanks to Andrew Ng, deeplearning.ai and Coursera for teaching me how to develop my own Deep Learning project.

<img src="https://github.com/enyangxxx/tfFoodImageClassifier/blob/master/gitImg/food1.jpg" width="200" height="200">

<img src="https://github.com/enyangxxx/tfFoodImageClassifier/blob/master/gitImg/food2.jpg" width="200" height="200">

<img src="https://github.com/enyangxxx/tfFoodImageClassifier/blob/master/gitImg/nonfood.jpg" width="200" height="200">
