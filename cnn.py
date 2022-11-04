# Alexander Cox
# Ernst-Richard Kausche
# Ava Sato
import sys
import random
import pandas as pd
from PIL import Image
import numpy as np


class Neuron:
    def __init__(self, w0, w, activation):
        self.w0 = w0
        self.w = w
        self.activation = activation


class CNN:
    def __init__(self, n_hidden, eta):
        print(f"Initializing model: {n_hidden}n, {eta}r...")
        # Assign arguments to attributes
        self.n_hidden = int(n_hidden)
        self.eta = float(eta)
        # Randomly assign hidden neuron weights to reals in [-0.1, 0.1) and set feedback and activation to 0
        self.hidden_neurons = []
        for i in range(self.n_hidden):
            w0 = (random.random() - 0.5) * 0.2
            weights = []
            for j in range(self.train_set.shape[1] - 1):
                weights.append((random.random() - 0.5) * 0.2)
            self.hidden_neurons.append(Neuron(w0, weights, 0))
        # Randomly assign output neuron weights to reals in [-0.1, 0.1) and set feedback and activation to 0
        w0 = (random.random() - 0.5) * 0.2
        weights = []
        for j in range(self.n_hidden):
            weights.append((random.random() - 0.5) * 0.2)
        self.output_neuron = Neuron(w0, weights, 0)

    def train(self, train_set, val_set):
        print("Training Model...")
        epoch = 0
        acc = 0
        test_list = []
        val_list = []
        # Run up to 500 epochs or stop if accuracy reaches 99%
        while epoch < 500 and acc < 0.99:
            # Run predictions and backpropogate for each instance in the test set
            test_acc = 0
            for i in range(train_set.shape[0]):
                # Feed instance forward through network
                # Calculate activation for each neuron j in the hidden layer using the sigmoid squashing function
                hidden_acts = []
                for j in range(self.n_hidden):
                    hn_j = self.hidden_neurons[j]
                    hn_j.activation = 1 / (1 + np.exp(-1 * (hn_j.w0 + np.dot(train_set[i, 1:], hn_j.w))))
                    hidden_acts.append(hn_j.activation)
                # Calculate activation of output neuron using each activation of each hidden neuron
                self.output_neuron.activation = \
                    1 / (1 + np.exp(-1 * (self.output_neuron.w0 + np.dot(hidden_acts, self.output_neuron.w))))
                # Backpropogate based on output neuron activation
                self.backprop(self.output_neuron.activation, train_set[i, 0], i)
                if round(self.output_neuron.activation) == train_set[i, 0]:
                    test_acc += 1
            test_list.append(test_acc / train_set.shape[0])
            # Run predictions on validation set to get accuracy
            correct = 0
            for i in range(val_set.shape[0]):
                # Feed instance forward through network
                # Calculate activation for each neuron j in the hidden layer
                hidden_acts = []
                for j in range(self.n_hidden):
                    hn_j = self.hidden_neurons[j]
                    hn_j.activation = 1 / (1 + np.exp(-1 * (hn_j.w0 + np.dot(val_set[i, 1:], hn_j.w))))
                    hidden_acts.append(hn_j.activation)
                # Calculate activation of output neuron using each activation of each hidden neuron
                self.output_neuron.activation = \
                    1 / (1 + np.exp(-1 * (self.output_neuron.w0 + np.dot(hidden_acts, self.output_neuron.w))))
                if round(self.output_neuron.activation) == val_set[i, 0]:
                    correct += 1
            # Calculate accuracy and increment epoch number
            acc = correct / val_set.shape[0]
            val_list.append(acc)
            epoch += 1
            print(f"\tEpoch {epoch} - Accuracy: {acc}", end='\r')
        return test_list, val_list

    def backprop(self, pred, act, inst, train_set):
        # Calculate output feedback as activation' * error
        fb_o = pred * (1 - pred) * (act - pred)
        # Update weights for output neuron as w = w - eta * -input * output feedback, where input for w0 = 1
        self.output_neuron.w0 += self.eta * fb_o
        for i in range(self.n_hidden):
            self.output_neuron.w[i] += self.eta * self.hidden_neurons[i].activation * fb_o
        # Update weights for each hidden neuron
        for i in range(self.n_hidden):
            hn_i = self.hidden_neurons[i]
            # Calculate feedback for hidden neuron i as activation' * weight from i to o * fb_o
            fb_i = hn_i.activation * (1 - hn_i.activation) * self.output_neuron.w[i] * fb_o
            # Update weights for hidden neurons as w = w - eta * -input * feedback_k, where input for w0 = 1
            hn_i.w0 += self.eta * fb_i
            for j in range(len(hn_i.w)):
                hn_i.w[j] += self.eta * train_set[inst, j + 1] * fb_i

    def test(self, test_set):
        print("Testing Model...")
        # Define confusion matrix as a Data Frame for easy conversion to csv later
        conf_mat = pd.DataFrame({0: [0, 0], 1: [0, 0]})
        correct = 0
        for i in range(test_set.shape[0]):
            # Feed instance forward through network
            # Calculate activation for each neuron j in the hidden layer
            hidden_acts = []
            for j in range(self.n_hidden):
                hn_j = self.hidden_neurons[j]
                hn_j.activation = 1 / (1 + np.exp(-1 * (hn_j.w0 + np.dot(test_set[i, 1:], hn_j.w))))
                hidden_acts.append(hn_j.activation)
            # Calculate activation of output neuron using each activation of each hidden neuron
            self.output_neuron.activation = \
                1 / (1 + np.exp(-1 * (self.output_neuron.w0 + np.dot(hidden_acts, self.output_neuron.w))))
            # Get prediction based on threshold hyperparameter
            pred = -1
            if self.output_neuron.activation >= self.thresh:
                pred = 1
            elif self.output_neuron.activation < self.thresh:
                pred = 0
            # Increment cell of confusion matrix that corresponds to the result
            conf_mat.loc[pred, test_set[i, 0]] += 1
            if pred == test_set[i, 0]:
                correct += 1
        # Save confusion matrix as csv file with format results_<DataSet>_<Neurons>n_<LearningRate>r_<Threshold>t
        # _<TrainingPercentage>p_<Seed>.csv
        conf_mat.to_csv(f'Results/results_{self.path[:len(self.path) - 4]}_'
                        f'{self.n_hidden}n_{self.eta}r_{self.thresh}t_{self.p_train}p_{self.seed}.csv')
        # Return overall accuracy of model on the test set
        print(f"\tTest Accuracy: {correct / test_set.shape[0]}")
        return correct / test_set.shape[0]


def process_data():
    # Set random seed
    random.seed(seed)
    # Create a set of numpy arrays for each image file in the data set
    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    train_set = []
    for genre in GENRES:
        for i in range(100):
            if i < 10:
                song_arr = np.array(Image.open(f'Data/images_original/{genre}/{genre}0000{i}.png'))[:, :, :3]
                if song_arr.shape != (288, 432, 3):
                    print(f'OOPSIES: {genre}0000{i}.png shape = {song_arr.shape}, should be (288, 432, 3)')
                train_set.append(song_arr)
            else:
                if genre != 'jazz' or i != 54:
                    song_arr = np.array(Image.open(f'Data/images_original/{genre}/{genre}000{i}.png'))[:, :, :3]
                    if song_arr.shape != (288, 432, 3):
                        print(f'OOPSIES: {genre}0000{i}.png shape = {song_arr.shape}, should be (288, 432, 3)')
                    train_set.append(song_arr)
    n_inst = len(train_set)
    # Generate validation set randomly from other half of non-training data
    val_set = []
    for i in range(int(((1 - train_p) / 2) * n_inst)):
        rand = random.randint(0, len(train_set) - 1)
        val_set.append(train_set[rand])
        train_set.pop(rand)
    # Generate test set randomly from half of non-training data
    test_set = []
    for i in range(int(((1 - train_p) / 2) * n_inst)):
        rand = random.randint(0, len(train_set) - 1)
        test_set.append(train_set[rand])
        train_set.pop(rand)
    return train_set, val_set, test_set


if __name__ == '__main__':
    seed = sys.argv[1]
    train_p = float(sys.argv[2])
    train_set, val_set, test_set = process_data()
    print(len(train_set))
