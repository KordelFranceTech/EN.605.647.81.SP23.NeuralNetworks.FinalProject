# PIP packages
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Local packages
from rbm.restricted_boltzmann_machine import RBM
from rbm.activation_functions import Sigmoid, ReLU, HyperTangent
from helper_functions import build_label, concat_label
import hyperparameters


def load_data(single_number=None, with_labels=True):
    """
    Load the MNIST dataset and obtain the specific number of interest
    :param single_number: int - the specific number to filter out of the dataset
    :param with_labels: bool - flag indicating to embed label into training data
    :return: np.array - the list of images
    """
    # mnist = fetch_openml('mnist_784')
    mnist = load_digits()
    input_data = mnist.data / 255.0
    labels = mnist.target
    if single_number is not None:
        input_data = input_data[labels == single_number]
        labels = [single_number] * len(input_data)
    labeled_input_data: list = []
    for i in range(0, len(input_data)):
        if single_number is not None:
            label = build_label(single_number, input_data[i].shape, with_labels)
        else:
            label = build_label(labels[i], input_data[i].shape, with_labels)
        labeled_input_data.append(np.concatenate((input_data[i], label)))

    final_data = np.array(labeled_input_data)
    print(f"input data shape: {final_data.shape}")
    return final_data


def build_rbm_for_number(number: int,
                         n_hidden: int,
                         epochs: int,
                         batch_size: int,
                         eta: float,
                         activation_fn,
                         save_initial_images: bool = True,
                         save_reconstructed_images: bool = True,
                         show_error_plots: bool = False):
    """
    Builds a Restricted Boltzmann Machine to learn a single number from the MNIST dataset.
    Plots and saves the reconstructed images.

    :param number: int - the specific number to filter out of the dataset
    :param save_initial_images: bool - flag indicating to save initial MNIST images
    :param save_reconstructed_images: bool - flag indicating to save reconstructed images from trained RBM
    :param show_error_plots: bool - flag indicating whether or not to show training error plots
    :return: null
    """
    # Load the MNIST data for the specified number only
    input_data = load_data(single_number=number, with_labels=True)
    rbm = RBM(
        n_hidden=n_hidden,
        epochs=epochs,
        batch_size=batch_size,
        eta=eta,
        act_fn=activation_fn
    )
    rbm.train_rbm(input_data=input_data, weights=None, biases=None, debug=False)

    # Plot training errors
    if show_error_plots:
        errors, = plt.plot(range(len(rbm.mean_squared_errors)), rbm.mean_squared_errors, label="Training Error")
        plt.legend(handles=[errors])
        plt.title(f"Error Plot for {number}, Size = {n_hidden}")
        plt.ylabel('MSE')
        plt.xlabel('Epochs')
        plt.show()

    # Reconstruct the images from the trained RBM
    reconstruction_data = load_data(single_number=number, with_labels=False)
    reconstructed_images = [
        rbm.reconstruct(reconstruction_data[
                            np.random.choice(range(reconstruction_data.shape[0]), rbm.batch_size)
                        ]) for i in range(16)
        ]

    # Save initial images that were input into the RBM
    if save_initial_images:
        fig, axs = plt.subplots(2, 2)
        plt.suptitle("Restricted Boltzmann Machine - Initial Unaltered MNIST images")
        im_count: int = 0

        # Sample 16 of the initial unaltered  MNIST images and display them
        for i in range(2):
            for j in range(2):
                axs[i, j].imshow(reconstruction_data[im_count].reshape((16, 8)), cmap='gray')
                axs[i, j].axis('off')
                im_count += 1
        fig.savefig(f"./images/number_{number}_initial.png")
        plt.close()

    # Save images reconstructed from the trained RBM
    if save_reconstructed_images:
        fig, axs = plt.subplots(2, 2)
        plt.suptitle(f"Reconstructed RBM Images, Size = {n_hidden}")
        im_count: int = 0

        # Sample 16 reconstructed images and display them
        for i in range(2):
            for j in range(2):
                axs[i, j].imshow(reconstructed_images[-1][im_count].reshape((16, 8)), cmap='gray')
                axs[i, j].axis('off')
                im_count += 1
        fig.savefig(f"./images/number_{number}_reconstructed_{n_hidden}_nodes.png")
        plt.close()

    return rbm, errors


def tune_hyperparameters():
    ##------------------------------------------------------------------------------------------------------------------
    # Declare hyperparameters to tune over
    rbm_sizes: list = [4,8,16,32,64,128,256,512]
    batch_sizes: list = [4,8,16,32]
    etas: list = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    act_fns: list = [Sigmoid(), ReLU(), HyperTangent()]

    ##------------------------------------------------------------------------------------------------------------------
    # Finetune RBM Size
    rbms: list = []
    errors: list = []
    TEST: str = "RBM Size"
    for n_hidden in rbm_sizes:
        rbm,e = build_rbm_for_number(number=None,
                                     n_hidden=n_hidden,
                                     epochs=hyperparameters.EPOCHS,
                                     batch_size=4,
                                     eta=0.01,
                                     activation_fn=Sigmoid(),
                                     save_initial_images=True,
                                     save_reconstructed_images=True,
                                     show_error_plots=True)
        rbms.append(rbm)
        errors.append(e)
    for i in range(0, len(rbms)):
        rbm = rbms[i]
        e = errors[i]
        plt.plot(range(len(rbm.mean_squared_errors)), rbm.mean_squared_errors, label="Training Error")
        plt.legend(handles=[e])
        plt.title(f"Error Plot for {TEST}")
        plt.ylabel('MSE')
        plt.xlabel('Epoch Count')
    plt.legend([f"{TEST} =  {i}" for i in etas])
    plt.show()

    ##------------------------------------------------------------------------------------------------------------------
    # Finetune Batch Size
    rbms.clear()
    errors.clear()
    TEST: str = "Batch Size"
    for batch_size in batch_sizes:
        rbm,e = build_rbm_for_number(number=None,
                                     n_hidden=64,
                                     epochs=hyperparameters.EPOCHS,
                                     batch_size=batch_size,
                                     eta=0.01,
                                     activation_fn=Sigmoid(),
                                     save_initial_images=True,
                                     save_reconstructed_images=True,
                                     show_error_plots=True)
        rbms.append(rbm)
        errors.append(e)
    for i in range(0, len(rbms)):
        rbm = rbms[i]
        e = errors[i]
        plt.plot(range(len(rbm.mean_squared_errors)), rbm.mean_squared_errors, label="Training Error")
        plt.legend(handles=[e])
        plt.title(f"Error Plot for {TEST}")
        plt.ylabel('MSE')
        plt.xlabel('Epoch Count')
    plt.legend([f"{TEST} =  {i}" for i in etas])
    plt.show()

    ##------------------------------------------------------------------------------------------------------------------
    # Finetune Learning Rate
    rbms.clear()
    errors.clear()
    TEST: str = "Learning Rate"
    for eta in etas:
        rbm,e = build_rbm_for_number(number=None,
                                     n_hidden=64,
                                     epochs=hyperparameters.EPOCHS,
                                     batch_size=4,
                                     eta=eta,
                                     activation_fn=Sigmoid(),
                                     save_initial_images=True,
                                     save_reconstructed_images=True,
                                     show_error_plots=True)
        rbms.append(rbm)
        errors.append(e)
    for i in range(0, len(rbms)):
        rbm = rbms[i]
        e = errors[i]
        plt.plot(range(len(rbm.mean_squared_errors)), rbm.mean_squared_errors, label="Training Error")
        plt.legend(handles=[e])
        plt.title(f"Error Plot for {TEST}")
        plt.ylabel('MSE')
        plt.xlabel('Epoch Count')
    plt.legend([f"{TEST} = {i}" for i in etas])
    plt.show()

    ##------------------------------------------------------------------------------------------------------------------
    # Finetune Activatin Function
    rbms.clear()
    errors.clear()
    TEST: str = "Activation Function"
    for act_fn in act_fns:
        rbm, e = build_rbm_for_number(number=None,
                                      n_hidden=64,
                                      epochs=hyperparameters.EPOCHS,
                                      batch_size=4,
                                      eta=0.01,
                                      activation_fn=act_fn,
                                      save_initial_images=True,
                                      save_reconstructed_images=True,
                                      show_error_plots=True)
        rbms.append(rbm)
        errors.append(e)
    for i in range(0, len(rbms)):
        rbm = rbms[i]
        e = errors[i]
        plt.plot(range(len(rbm.mean_squared_errors)), rbm.mean_squared_errors, label="Training Error")
        plt.legend(handles=[e])
        plt.title(f"Error Plot for {TEST}")
        plt.ylabel('MSE')
        plt.xlabel('Epoch Count')
    plt.legend([f"{TEST} = {i}" for i in etas])
    plt.show()

    ##------------------------------------------------------------------------------------------------------------------
    # Set Final Tuning Parameters
    hyperparameters.NUM_HIDDEN_NODES = 64
    hyperparameters.BATCH_SIZE = 4
    hyperparameters.ETA = 0.01
    hyperparameters.ACTIVATION_FUNCTION = Sigmoid()
    hyperparameters.EPOCHS = 300

    ##------------------------------------------------------------------------------------------------------------------
    # Build Final Model from Tuned Parameters
    acc, e = build_rbm_for_number(number=None,
                                  n_hidden=hyperparameters.NUM_HIDDEN_NODES,
                                  epochs=hyperparameters.EPOCHS,
                                  batch_size=hyperparameters.BATCH_SIZE,
                                  eta=hyperparameters.ETA,
                                  activation_fn=hyperparameters.ACTIVATION_FUNCTION,
                                  save_initial_images=True,
                                  save_reconstructed_images=True,
                                  show_error_plots=True)

