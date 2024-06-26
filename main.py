import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
from Network import *
from sklearn.metrics import mean_absolute_error

melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)
Price_Converter = np.max(melbourne_data.Price) - np.min(melbourne_data.Price)

y = preprocessing.minmax_scale(melbourne_data.Price)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(melbourne_data[melbourne_features])

train_X, val_x, train_y, val_y = train_test_split(X, y, random_state=0)
train_y = np.reshape(train_y, (-1,1))
val_y = np.reshape(val_y, (-1,1))

# Batch Size
M = len(train_X)
# M = 32

model = Network([len(melbourne_features), 10, 1])
data_errors = []
test_errors = []
EPOCHS = 1000
for _ in range(EPOCHS):
    # X is shape 4647 x 7
    batch = np.random.randint(train_X.shape[0], size=M)
    activation, Z = model.forward(train_X[batch, :].T)
    test_prediction, _ = model.forward(val_x.T)
    data_errors.append(mean_absolute_error(train_y[batch, :].T, activation[-1]))
    test_errors.append(mean_absolute_error(val_y.T, test_prediction[-1]))
    
    print(mean_absolute_error(train_y[batch, :].T, activation[-1])*Price_Converter)
    model.train(activation, Z, train_y[batch, :].T, M, train_X[batch, :])
    # M+=1

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(EPOCHS), data_errors, color='tab:orange', label='Train Error')
ax.plot(range(EPOCHS), test_errors, color='tab:blue', label='Test Error')

ax.set_xlabel('Epochs')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Training and Validation Error over Epochs')
ax.legend()

plt.show()



# 'Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
# 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
# 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
# 'Longtitude', 'Regionname', 'Propertycount'

#345928.787508133
#301205.2524428287
#268016.3623971688