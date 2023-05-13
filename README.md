## Problem
To design a neural network that classify small squared boxes, precisely VIN character boxes.
## Used data
The EMNIST dataset is a set of a 28x28 pixel images containing handwritten letters and digits. We consider the Balanced split that contains 112,800 train and 18,800 test images. Because of our specific problem, we drop classes that correspond to lowercase letters and also uppercase "I", "O", "Q" (They are not used in VIN).
## Methods & Ideas & Accuracy
Experiments were done with classic CNN architectures, ResNets and eventually Ensemble. Too deep networks performed worse, so shallower ones were considered as main objective. We regard a VGG-like model with 4 convolutional layers that showed accuracy of 91% on test set before applying augmentation and ResNet18 that performed almost equally, despite the higher depth and complexity. The models were overfitting easily, so augmentation was applied. After that models started converging extremely slowly and inconsistently, but they weren't overfitting that much anymore. To speed up the process the Ensemble model was implemented by taking outputs of our VGG-like model and ResNet18 before last layer, adding them and proceeding through a fully connected layer. Performance of this model was much higher (about 90% accuracy just from the start) and at last it became converging to the point where it can't proceed any further with constant learning rate, so scheduler was applied and eventually the perfomance of the model increased by ~1.5%, ending up at the point of ~96% accuracy on test set where nothing could help.
## Usage
Make sure that your images are in a folder named **test_data**, which is in the same directory as the inference.py script. 
You can test the script by building a docker image and executing the following command:
```
docker run -i <docker image> python inference.py --input test_data
```
or even more simply:
```
docker run <docker image>
```
Output:
```
72,test_data/72_nveErw.png
49,test_data/49_GclrwW.png
85,test_data/85_cFuoLx.png
68,test_data/68_MfpKBw.png
49,test_data/76_ZAXstB.png
77,test_data/77_EGaGVu.png
82,test_data/82_huccLv.png
83,test_data/83_IoqYlBf.jpg
75,test_data/75_ObCkec.png
```

