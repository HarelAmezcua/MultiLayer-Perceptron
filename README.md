Multi-Layer Perceptron for XOR Function

This repository contains an implementation of a simple Multi-Layer Perceptron (MLP) network designed to learn the behavior of an XOR function with two inputs. The network employs back-propagation for learning and demonstrates the basic principles of neural networks, including forward propagation, error calculation, and the back-propagation of this error to update the weights and biases.

Project Structure
simple_mlp.m: The main MATLAB script that implements the MLP, including the training and evaluation of the network.
Features
Implementation of a multi-layer perceptron with one hidden layer.
Utilizes the sigmoid activation function and its derivative for back-propagation.
Demonstrates the network's ability to learn the XOR function.
Includes the calculation of the mean squared error (MSE) to evaluate the model's performance.
Requirements
To run this project, you will need:

MATLAB (The code was developed and tested in MATLAB, but it might also run in GNU Octave with minor modifications.)
Running the Code
Clone the repository to your local machine.
Open MATLAB and navigate to the directory containing the cloned repository.
Run the script simple_mlp.m by typing simple_mlp in the MATLAB command window.
Methodology
The MLP consists of an input layer, one hidden layer, and an output layer. The network learns to approximate the XOR function through the back-propagation learning algorithm. The script initializes the network's weights randomly and updates these weights iteratively to minimize the error between the predicted and actual outputs.

Training Process
The network is trained on the four possible inputs of the XOR function.
The weights are updated using the gradient descent optimization method.
The learning rate and the number of epochs (iterations) are adjustable parameters.
Evaluation
The script evaluates the network's performance by plotting the error and mean squared error over epochs.
The final weights and biases are outputted after training, demonstrating the network's learned parameters.
Results and Conclusion
The repository includes a detailed report of the code execution, encompassing graphs of the function to be approximated, the approximate function, the plot of the error, and the mean squared error. Additionally, it discusses the generalization capability of the network and provides conclusions and observations based on the training and evaluation results.

For detailed results and further insights, please refer to the code comments and inline documentation.

Contributing
Feel free to fork the repository, make changes, and submit pull requests if you have ideas for improvements or have found bugs. For major changes, please open an issue first to discuss what you would like to change.

