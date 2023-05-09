# RBM Project - Scenario 2

___
This project was built for the Neural Networks course at Johns Hopkins University, EN.605.647.81.

It facilitates the construction and execution of a Restricted Boltzmann Machine Classifier by first learning the 
complexity of the data to determine the optimal number of hidden nodes needed to learn each digit class of the MNIST
dataset, and then using that information to construct an RBM that can classify all ten MNIST digits. Once this RBM
is constructed, we fine-tune the learning rate, batch size, and activation function to learn the optimal hyperparameters.
___
To run:


1) `cd` into `RBMProject` directory:
```python
    cd RBMProject
```

2) Set hyperparameters in `hyperparameters.py` (default parameters are already set, so you can leave them as is for demo purposes).


3) Run the program:
```python
    python main.py
```