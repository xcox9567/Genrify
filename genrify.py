# Authors: Alexander Cox, Ernst-Richard Kausche, Ava Sato

import sys
import random
import pandas as pd
from PIL import Image
import numpy as np

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
IMG_SIZE = (288, 432, 3)


class MLP:
    """
        A class representing a multilayer perceptron
          - n_inputs: Integer representing the number of inputs going into multilayer perceptron
          - n_hidden: Integer representing the number of neurons to include in the hidden layer of the neural net
          - eta: Float representing the learning rate to use for stochastic gradient descent when updating weights
    """
    class Neuron:
        """
            A class representing an individual neuron in a multilayer perceptron
              - w0: Float representing the bias or weight 0 of the neuron
              - w: Set of floats representing the set of weights for the neuron
              - act: Float between -1 and 1 representing the activation of the neuron
        """
        def __init__(self, w0, w, act):
            self.w0 = w0
            self.w = w
            self.act = act

    def __init__(self, n_inputs, n_hidden, eta):
        print(f"Initializing model: {n_hidden}n, {eta}r...")
        # Assign arguments to attributes
        self.n_hidden = int(n_hidden)
        self.eta = float(eta)
        # Randomly assign hidden neuron weights to floats in [-0.1, 0.1) and set feedback and activation to 0
        self.hidden_neurons = []
        for i in range(self.n_hidden):
            w0 = (random.random() - 0.5) * 0.2
            weights = []
            for j in range(n_inputs):
                weights.append((random.random() - 0.5) * 0.2)
            self.hidden_neurons.append(MLP.Neuron(w0, weights, 0))
        # Randomly assign output neuron weights to floats in [-0.1, 0.1) and set feedback and activation to 0
        self.output_neurons = []
        for i in range(len(GENRES)):
            w0 = (random.random() - 0.5) * 0.2
            weights = []
            for j in range(self.n_hidden):
                weights.append((random.random() - 0.5) * 0.2)
            self.output_neurons.append(MLP.Neuron(w0, weights, 0))


def process_data():
    """
        Processes GTZAN spectrogram images into 3D numpy arrays and separates it into training, validation, and testing
        sets based on train_p hyperparameter.

        Returns a 3-tuple of the form (training data, validation data, testing data)
    """
    # Set random seed
    random.seed(seed)
    # Create a set of numpy arrays for each image file in the data set and store it in the train_set list
    train_set = []
    for genre in GENRES:
        for i in range(100):
            # If file number is less than 10 include an extra 0 in the file name
            if i < 10:
                # Convert image data to 3D numpy array
                song_arr = np.array(Image.open(f'Data/images_original/{genre}/{genre}0000{i}.png'))[:, :, :3]
                # If array isn't of a consistent size print a warning, and otherwise add data to array
                if song_arr.shape != IMG_SIZE:
                    print(f'OOPSIES: {genre}0000{i}.png shape = {song_arr.shape}, should be (288, 432, 3)')
                else:
                    train_set.append(song_arr)
            # If file number is not less than 10 remove the extra 0 from the file name
            else:
                # Exclude Data/images_original/jazz/jazz00054.png as it does not exist
                if genre != 'jazz' or i != 54:
                    # Convert image data to 3D numpy array
                    song_arr = np.array(Image.open(f'Data/images_original/{genre}/{genre}000{i}.png'))[:, :, :3]
                    # If array isn't of a consistent size print a warning, and otherwise add data to array
                    if song_arr.shape != IMG_SIZE:
                        print(f'OOPSIES: {genre}000{i}.png shape = {song_arr.shape}, should be (288, 432, 3)')
                    else:
                        train_set.append(song_arr)
    # Generate validation set randomly from half of non-training data
    val_set = []
    for i in range(int(((1 - train_p) / 2) * len(train_set))):
        rand = random.randint(0, len(train_set) - 1)
        val_set.append(train_set[rand])
        train_set.pop(rand)
    # Generate test set randomly from half of non-training data
    test_set = []
    for i in range(int(((1 - train_p) / 2) * len(train_set))):
        rand = random.randint(0, len(train_set) - 1)
        test_set.append(train_set[rand])
        train_set.pop(rand)
    return train_set, val_set, test_set


if __name__ == '__main__':
    seed = sys.argv[1]
    train_p = float(sys.argv[2])
    train_data, val_data, test_data = process_data()
    print(len(train_data))
