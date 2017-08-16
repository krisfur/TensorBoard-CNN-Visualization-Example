# TensorBoard CNN Visualization Example

This example Convolutional Neural Network using MNIST data was written to showcase usage of the TensorBoard visualisation tools. 
No functions are defined so code can be easily read from top to bottom, step by step.
Made for learning purposes.

## Getting Started

### Prerequisites
```
Python 3.x
```
```
TensorFlow 1.2.1 or higher 
```
Doesn't require GPU support, can be done on CPU only. 
If using GPU support you will need the CUPTI library for TensorBoard to work.

### Running the program

Simply execute:
```
python CNN_TB_Example.py
``` 
To train the CNN and get logs saved to mnist_TB_logs folder.
To view logs simply run:
```
tensorboard --logdir=mnist_TB_logs
```
In the command line/terminal while in the folder with the program. 
You can naturally replace "mnist_TB_logs" with path to any folder that contains TensorBoard logs.

## Built With

* [TensorFlow](https://www.tensorflow.org/) - Framework version 1.2.1 used

## Contributing

Feel free to contribute by forking and submitting a merge request.

## Authors

* **Krzysztof Furman** - *Initial work* - [krisfur](https://github.com/krisfur)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Google's [MNIST tutorials](https://www.tensorflow.org/get_started/mnist/pros) which were used as base for the CNN in this file.
* Ideas expressed in [this GitHub discussion](https://github.com/tensorflow/tensorflow/issues/842)
