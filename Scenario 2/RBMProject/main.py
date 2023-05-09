# Local packages
import rbm_tuning
import rbm_learn_architecture


if __name__ == "__main__":
    # Build 10 RBMs for each of the 10 MNIST numbers to learn data complexity
    rbm_learn_architecture.learn_digits()

    # Use the knowledge learned from data complexity to tune a final RBM
    rbm_tuning.tune_hyperparameters()

