import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt

import numpy as np
from Network import *
import copy

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
import torch.optim as optim
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
test_y = np.reshape(test_y, (-1,1))
train_y = np.reshape(train_y, (-1,1))
val_y = np.reshape(y, (-1,1))

train_data = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
# Batch Size
batch_size = 128

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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
    

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
# print("Done!")
tensor_X = torch.tensor(X, dtype=torch.float32).to(device)
tensor_y = torch.tensor(val_y).to(device)
print(tensor_X.shape, tensor_y.shape)
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
            # print(i, j)
            if min_loss < final_loss:
                original_state_dict['linear_relu_stack.0.weight'][j, i] = torch.tensor(best_weight, dtype=torch.float32)
                model.linear_relu_stack[0].weight.data[j, i] = best_weight
                data_errors.append(min_loss)
                # final_loss = min_loss
            best_weights.append(best_weight)
            lowest_error.append(min_loss)
            weights_errors.append(weight_error)
    for j in range(10):
        for i in range(10):
            weight_error = []
            # Create a copy of the model
            model_copy = copy.deepcopy(original_state_dict)
            min_loss = final_loss
            best_weight = 0
            for weight in weights:
                # Update the weight in the model copy
                model_copy['linear_relu_stack.2.weight'][j, i] = torch.tensor(weight, dtype=torch.float32)
                
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
            # print(i, j)
            if min_loss < final_loss:
                original_state_dict['linear_relu_stack.2.weight'][j, i] = torch.tensor(best_weight, dtype=torch.float32)
                model.linear_relu_stack[2].weight.data[j, i] = best_weight
                data_errors.append(min_loss)
                # final_loss = min_loss
            best_weights.append(best_weight)
            lowest_error.append(min_loss)
            weights_errors.append(weight_error)
    for j in range(10):
        for i in range(10):
            weight_error = []
            # Create a copy of the model
            model_copy = copy.deepcopy(original_state_dict)
            min_loss = final_loss
            best_weight = 0
            for weight in weights:
                # Update the weight in the model copy
                model_copy['linear_relu_stack.4.weight'][j, i] = torch.tensor(weight, dtype=torch.float32)
                
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
            # print(i, j)
            if min_loss < final_loss:
                original_state_dict['linear_relu_stack.4.weight'][j, i] = torch.tensor(best_weight, dtype=torch.float32)
                model.linear_relu_stack[4].weight.data[j, i] = best_weight
                data_errors.append(min_loss)
                # final_loss = min_loss
            best_weights.append(best_weight)
            lowest_error.append(min_loss)
            weights_errors.append(weight_error)
    weights = np.linspace(-1, 1, 20)
    for j in range(1):
        for i in range(10):
            weight_error = []
            # Create a copy of the model
            model_copy = copy.deepcopy(original_state_dict)
            min_loss = final_loss
            best_weight = 0
            for weight in weights:
                # Update the weight in the model copy
                model_copy['linear_relu_stack.6.weight'][j, i] = torch.tensor(weight, dtype=torch.float32)
                
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
            # print(i, j)
            if min_loss < final_loss:
                original_state_dict['linear_relu_stack.6.weight'][j, i] = torch.tensor(best_weight, dtype=torch.float32)
                model.linear_relu_stack[6].weight.data[j, i] = best_weight
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
gradient_loss_over_time = []
for _ in range(50):
    epochs = 10

    gradient_loss = loss_fn(model(tensor_X), tensor_y).item()
    gradient_loss_over_time.append(gradient_loss)
    _model, weights_errors = scenic_descendant(tensor_X, tensor_y, model)
    weights_errors_over_time.append(weights_errors)
    model = _model

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Old Error:",gradient_loss*Price_Converter," New error:", loss_fn(model(tensor_X), tensor_y).item()*Price_Converter)
print(min(gradient_loss_over_time)*Price_Converter)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
# Initial plot setup
ax.plot(range(len(data_errors)), data_errors, color='tab:orange', label='Train Error')
# def update_graph(val):
#     ax.clear()  # Clear previous plot
#     val = int(val)
#     for i in range(len(weights_errors_over_time[val])):
#         ax.plot(range(len(weights_errors_over_time[val][i])), weights_errors_over_time[val][i], label=f'Row {i+1}', color=colors[i % len(colors)])
#     ax.axhline(y=gradient_loss_over_time[val], color='purple', linestyle=':', linewidth=2, label='Gradient Loss')
    
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Mean Absolute Error')
#     ax.set_title('Training and Validation Error over Epochs')
#     ax.legend()

# # Initialize the plot with the first index
# update_graph(0)

# # Slider setup
# axindex = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# index = Slider(axindex, 'Index', 0, len(weights_errors_over_time) - 1, valinit=0, valstep=1)
# index.on_changed(update_graph)

plt.show()

# ax.plot(range(len(data_errors)), data_errors, color='tab:orange', label='Train Error')
# ax.plot(range(len(test_errors)), test_errors, color='tab:blue', label='Test Error')


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
#Old Error: 6959.854752580062  New error: 6926.378013475117
