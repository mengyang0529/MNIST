# Generate Handwrite Digit from MNIST 
* To generate an image of a handwrite digits sequence from MNIST
* Data augmentation tool for training model
* [MNIST database](http://yann.lecun.com/exdb/mnist/)

## Requirement
- python 3.7
- opencv 3.4.2
- numpy  1.15.1
- pillow 5.2.0 

## API
```python
def generate_numbers_sequence(digits, spacingRange, imageWidth):
    """
    Parameters
    ----------
    digits: 
        List of digits you want to generate(for example [2, 0, 1, 8, 0, 9, 0, 7]).

    spacingRange: 
        Tuple of spacing range(minimum, maximum). 
        The spacing (in pixel units) between digits is from minimum to maximum randomly.

    imageWidth:
        Width of the image in pixels. The widthValue > digitNum * 28. 28 is the data cols and rows of MNIST.
    
    return:
        Image results.
    -------
    Results 
        A floating point 32bit image of digits sequence, which scale ranging from 0 (black) to 1 (white).
```
```python
def random_test() #test generate_numbers_sequence() only
```
```python
def data_augmentation(option, subOption, parameters):
    """"
    Parameters
    ----------
    option: 
        Two processing ways have been implemented: Filter and Enhance. 

    subOption: 
        Several sub method for each option.
        Filter: GaussianBlur,BoxBlur,MaxFilter,MinFilter.
        Enhance: Contrast,Brightness,Sharpness.

    parameters:
        parameters for each subOption method.
    
    return:
        Image results.
    -------
    Results 
        Saving in the folder of PNGs/Filter and PNGs/Enhance.
        Format of image name: "digit label"_"index in MNIST"_"augmentation method"_"parameter"

```
## Note
- Basicly, data augmentation is to generate a diversity dataset.
  Including different shapes, colors, luminous intensities and so on, in order to cover all of the situations might happen.
- For different task (detection, recognition, segmentation) the way of data augmentation is different.
  Different dataset should be treat in proper way.
  For example for detection and regression random crop is a general method for data augmentation. 
  But in case of MNIST random crop is not proper, because of small size. And also MNIST is proper for recognition.
  Random crop will increase the complexity of recognition model training.
- Function of "data_augmentation" has not been perfect yet. Some of the methods have not included.
  It is just a simple frame to show basic ideas of data augmentation.
- No parameter recommendation for all of the methods. Generated data need to be check according to different parameters.
  Only select proper parameters or image data according to your purpose of task.
  For example, in case you decide to use a pre-processing, too much diversity may increase the complexity to train the model.
  You just need to select data similar with output of pre-processing step.
  But if the environment (luminous intensities) is not so complex. 
  Without pre-processing but to train your model using enough data could save time.




