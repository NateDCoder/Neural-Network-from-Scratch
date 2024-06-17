import numpy as np

class Network():
    def __init__(self, layer_sizes):
        self.LR = 0.01
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            # L mean layer length and m is batch side and it showing the matrix multiplaction
            # L+1 x L * L x M
            print(layer_sizes[i+1], layer_sizes[i])
            self.weights.append(np.random.uniform(low=-1, high=1, size=(layer_sizes[i+1], layer_sizes[i])))
            self.biases.append(np.random.uniform(low=-1, high=1, size=(layer_sizes[i+1], 1)))
    def forward(self, X):
        activation_layers = []
        Z = []
        Z.append((self.weights[0] @ X)+self.biases[0])
        activation_layers.append(np.maximum(Z[0], 0))
        Z.append((self.weights[1] @ activation_layers[0])+self.biases[1])
        activation_layers.append(1 / (1 + np.exp(-Z[1])))
        return activation_layers, Z

    def train(self, activation, Z, output, M, X):

        derative_output_error = 2*(activation[-1] - output)
        weight_delta_2 = (1/M) * (derative_output_error @ activation[0].T)
        bias_delta_2 = (1/M)*np.sum(derative_output_error, axis=1, keepdims=True)
        # bias_delta.append((1/M)*np.sum(derative_output_error, axis=1, keepdims=True))
            
        derivative_of_relu = np.where(Z[0] > 0, 1, 0)
        hiden_layer_error = (self.weights[1].T @ derative_output_error) * derivative_of_relu
        # if layer_count > 2:
        #     for i in range(layer_count-2):
        #         i = layer_count-2-i
        #         weight_delta.insert(0, (1/M) * (hiden_layer_error@self.Z[i-1]))
        #         bias_delta.insert(0, (1/M) * np.sum(hiden_layer_error, axis=1, keepdims=True))

        #         derivative_of_relu = np.where(self.Z[i-1] > 0, 1, 0)
        #         hiden_layer_error = (self.weights[i].T @ derative_output_error) * derivative_of_relu
        # print(hiden_layer_error.shape)
        weight_delta_1 = (1/M) * (hiden_layer_error@X)
        # weight_delta.insert(0, (1/M) * (hiden_layer_error@X))
        bias_delta_1 = (1/M) * np.sum(hiden_layer_error, axis=1, keepdims=True)
        # bias_delta.insert(0, (1/M) * np.sum(hiden_layer_error, axis=1, keepdims=True))
        
        self.weights[0] -= weight_delta_1
        self.weights[1] -= weight_delta_2

        self.biases[0] -= bias_delta_1
        self.biases[1] -= bias_delta_2

        # for i in range(layer_count):
        #     print(self.weights[i].shape, weight_delta[i].shape)
        #     self.weights[i] -= weight_delta[i]
        #     self.biases[i] -= bias_delta[i]
    