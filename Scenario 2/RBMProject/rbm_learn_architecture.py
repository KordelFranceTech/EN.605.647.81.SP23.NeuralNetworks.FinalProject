# PIP packages
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Local packages
from rbm.restricted_boltzmann_machine import RBM
from helper_functions import build_label, concat_label
import hyperparameters


def load_data(single_number=None, with_labels=True):
    """
    Load the MNIST dataset and obtain the specific number of interest
    :param single_number: int - the specific number to filter out of the dataset
    :return: np.array - the list of images
    """
    mnist = load_digits()
    input_data = mnist.data / 255.0
    labels = mnist.target
    if single_number is not None:
        input_data = input_data[labels == single_number]
        labels = [single_number] * len(input_data)
    labeled_input_data: list = []
    for i in range(0, len(input_data)):
        if single_number is not None:
            labeled_data = concat_label(single_number, input_data[i], with_labels)
        else:
            labeled_data = concat_label(labels[i], input_data[i], with_labels)
        labeled_input_data.append(labeled_data)

    final_data = np.array(labeled_input_data)
    print(f"input data shape: {final_data.shape}")
    return final_data


def build_rbm_for_number(number: int,
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
    # input_data = load_data_for_single_digit(number=number)
    input_data = load_data(single_number=number, with_labels=True)
    rbm = RBM(
        n_hidden=hyperparameters.NUM_HIDDEN_NODES,
        epochs=hyperparameters.EPOCHS,
        batch_size=hyperparameters.BATCH_SIZE,
        eta=hyperparameters.ETA,
        act_fn=hyperparameters.ACTIVATION_FUNCTION
    )
    rbm.train_rbm(input_data=input_data, weights=None, biases=None, debug=False)

    # Plot training errors
    if show_error_plots:
        errors, = plt.plot(range(len(rbm.mean_squared_errors)), rbm.mean_squared_errors, label="Training Error")
        plt.legend(handles=[errors])
        plt.title(f"Error Plot for {number}")
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
        fig, axs = plt.subplots(4, 4)
        plt.suptitle("Restricted Boltzmann Machine - Initial Unaltered MNIST images")
        im_count: int = 0

        # Sample 16 of the initial unaltered  MNIST images and display them
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(reconstruction_data[im_count].reshape((8, 8)), cmap='gray')
                axs[i, j].axis('off')
                im_count += 1
        fig.savefig(f"./images/number_{number}_initial.png")
        plt.close()

    # Save images reconstructed from the trained RBM
    if save_reconstructed_images:
        fig, axs = plt.subplots(4, 4)
        plt.suptitle("Restricted Boltzmann Machine - Reconstructed RBM Images")
        im_count: int = 0

        # Sample 16 reconstructed images and display them
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(reconstructed_images[-1][im_count].reshape((8, 8)), cmap='gray')
                axs[i, j].axis('off')
                im_count += 1
        fig.savefig(f"./images/number_{number}_reconstructed.png")
        plt.close()

    return rbm


def learn_digits():
    # Build 10 RBMs for each of the 10 MNIST numbers
    for i in range(0, 10):
        acc = build_rbm_for_number(number=i,
                                   save_initial_images=True,
                                   save_reconstructed_images=True,
                                   show_error_plots=True)
