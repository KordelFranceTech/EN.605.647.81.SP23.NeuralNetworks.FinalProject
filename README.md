# EN.605.647.81.SP23.NeuralNetworks.FinalProject
### Authors: Brett Wolff, Kordel France, and Hunter Klamut
___
This project was built for the Neural Networks course at Johns Hopkins University, EN.605.647.81. 

The code here pertains to the experiments and data presented in the corresponding research paper for the final project.

The folder `Scenario 1` contains all code and experiments pertaining to the data and results presented in the paper 
corresponding to `Scenario 1` where a multi-class Restricted Boltzmann Machine classifier is constructed and its 
performance analyzed over the MNIST dataset. We asses the loss and performance of a RBM to work as a classifier and 
reconstruct its training data.

The folder `Scenario 2` contains all code and experiments pertaining to the data and results presented in the paper 
corresponding to `Scenario 2` where we construct and execute a Restricted 
Boltzmann Machine classifier by first learning the complexity of the data to determine the optimal number of hidden 
nodes needed to learn each digit class of the MNIST dataset. We then use that information to construct an RBM that 
can classify all ten MNIST digits. Once this RBM is constructed, we fine-tune the learning rate, batch size, and activation 
function to learn the optimal hyperparameters.

Instructions to run the code pressented in each scenario are contained within each scenario folder.
___
