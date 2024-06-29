import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from Network import *
from sklearn.metrics import mean_absolute_error
import copy

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)
Price_Converter = np.max(melbourne_data.Price) - np.min(melbourne_data.Price)

y = preprocessing.minmax_scale(melbourne_data.Price)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(melbourne_data[melbourne_features])

train_X, test_x, train_y, test_y = train_test_split(X, y, random_state=0)
train_y = np.reshape(train_y, (-1,1))
val_y = np.reshape(test_y, (-1,1))

train_data = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
# Batch Size
batch_size = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
data_errors = []
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    counter = 0
    total_error = 0
    for batch, (X, y) in enumerate(dataloader):
        if len(X) < batch_size:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            break
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_error += loss_fn(pred, y).item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        counter+=1
    data_errors.append((total_error/counter))
test_errors = []           
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {Price_Converter*test_loss:>8f} \n")
    test_errors.append(Price_Converter*test_loss)
    

# epochs = 50
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")
tensor_X = torch.tensor(train_X, dtype=torch.float32).to(device)
tensor_y = torch.tensor(train_y).to(device)

def scenic_descendant(tensor_X, tensor_y, model):
    weights = np.linspace(-3.5, 3.5, 20)  # 2000 weights from -1 to 1

    # Container to store the errors
    weights_errors = []
    # Save the original model state
    original_state_dict = copy.deepcopy(model.state_dict())
            
    # Compute the loss and store it
    final_loss = loss_fn(model(tensor_X), tensor_y).item()
    best_weights = []
    lowest_error = []
    # Iterate through the weight values
    for j in range(10):
        for i in range(7):
            weight_error = []
            # Create a copy of the model
            model_copy = copy.deepcopy(original_state_dict)
            min_loss = final_loss
            best_weight = 0
            for weight in weights:
                # Update the weight in the model copy
                model_copy['linear_relu_stack.0.weight'][j, i] = torch.tensor(weight, dtype=torch.float32)
                
                # Load the modified state_dict into the model
                model.load_state_dict(model_copy)

                # Make predictions
                pred = model(tensor_X)
                
                # Compute the loss and store it
                loss = loss_fn(pred, tensor_y).item()
                if loss < min_loss:
                    min_loss = loss
                    best_weight = weight
                weight_error.append(loss)
            print(i, j)
            if min_loss < final_loss:
                original_state_dict['linear_relu_stack.0.weight'][j, i] = torch.tensor(best_weight, dtype=torch.float32)
                model.linear_relu_stack[0].weight.data[j, i] = best_weight
                data_errors.append(min_loss)
                # final_loss = min_loss
            best_weights.append(best_weight)
            lowest_error.append(min_loss)
            weights_errors.append(weight_error)
    return model, weights_errors
# for i in range(len(best_weights)):
#     if lowest_error[i] < final_loss:
#         model.linear_relu_stack[0].weight.data[0, i] = best_weights[i]
# min_index = lowest_error.index(min(lowest_error))
# print(min_index, best_weights[min_index], model.linear_relu_stack[0].weight.data[0, min_index], lowest_error[min_index])
# model.linear_relu_stack[0].weight.data[0, min_index] = best_weights[min_index]

# Change of the weight graph
weights_errors_over_time = []
for _ in range(10):
    epochs = 3

    gradient_loss = loss_fn(model(tensor_X), tensor_y).item()
    _model, weights_errors = scenic_descendant(tensor_X, tensor_y, model)
    weights_errors_over_time.append(weights_errors)
    model = _model

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Old Error:",gradient_loss*Price_Converter," New error:", loss_fn(model(tensor_X), tensor_y).item()*Price_Converter)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

def plot_graph(index, graph_type):
    plt.figure(figsize=(15, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Red, Green, Blue, Cyan, Magenta, Yellow, Black

    for i in range(len(weights_errors_over_time[index])):
        if graph_type == 'Line':
            plt.plot(range(len(weights_errors_over_time[index][i])), weights_errors_over_time[index][i], 
                     label=f'Row {i+1}', color=colors[i % len(colors)])
        elif graph_type == 'Scatter':
            plt.scatter(range(len(weights_errors_over_time[index][i])), weights_errors_over_time[index][i], 
                        label=f'Row {i+1}', color=colors[i % len(colors)])
        elif graph_type == 'Bar':
            plt.bar(range(len(weights_errors_over_time[index][i])), weights_errors_over_time[index][i], 
                    label=f'Row {i+1}', color=colors[i % len(colors)])

    plt.axhline(y=gradient_loss, color='purple', linestyle=':', linewidth=2)
    plt.legend(loc='upper right')
    plt.title(f"Graph for Index {index} with {graph_type} Plot")
    plt.show()

# ax.plot(range(len(data_errors)), data_errors, color='tab:orange', label='Train Error')
# ax.plot(range(len(test_errors)), test_errors, color='tab:blue', label='Test Error')

# List of colors (you can specify more if needed)
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Red, Green, Blue, Cyan, Magenta, Yellow, Black

# # Plot each row with specific colors
# for i in range(len(weights_errors_over_time[-1])):
#     plt.plot(range(len(weights_errors_over_time[-1][i])), weights_errors_over_time[-1][i], label=f'Row {i+1}', color=colors[i % len(colors)])
#     plt.axhline(y=gradient_loss, color='purple', linestyle=':', linewidth=2)
# Slider for selecting the index of the graph
index_slider = widgets.IntSlider(value=0, min=0, max=69, step=1, description='Index')

# Dropdown for selecting the graph type
graph_type_dropdown = widgets.Dropdown(
    options=['Line', 'Scatter', 'Bar'],
    value='Line',
    description='Graph Type'
)

# Display the widgets and the output
widgets.interact(plot_graph, index=index_slider, graph_type=graph_type_dropdown)
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Mean Absolute Error')
# ax.set_title('Training and Validation Error over Epochs')
# ax.legend()

# plt.show()


# 'Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
# 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
# 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
# 'Longtitude', 'Regionname', 'Propertycount'

#345928.787508133
#301205.2524428287
#268016.3623971688
#93005.8247976750
#94625.889353

#Old Error: 46052.209438548954  New error: 33440.8756093052
#Old Error: 49877.73207556858  New error: 41715.3976768787
#Old Error: 58540.399050158376  New error: 40663.464984242826
#Old Error: 37463.00860548409  New error: 32475.756490965367
#Old Error: 59218.8676921746  New error: 30499.44054692157

#Old Error: 54321.240147621225  New error: 23531.23441310866
#Old Error: 38236.379775062735  New error: 18533.48115359793
