import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np

melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)
Price_Converter = np.max(melbourne_data.Price) - np.min(melbourne_data.Price)

y = melbourne_data.Price
sorted_indexs = np.argsort(y)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']


scaler = preprocessing.StandardScaler()
X = melbourne_data[melbourne_features]
# X = scaler.fit_transform(melbourne_data[melbourne_features])
print(X.shape)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.sort(y),np.array(X['Landsize'])[sorted_indexs], color='tab:orange', label='Train Error')
# ax.plot(range(EPOCHS), test_errors, color='tab:blue', label='Test Error')

ax.set_xlabel('Rooms')
ax.set_ylabel('Price')
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