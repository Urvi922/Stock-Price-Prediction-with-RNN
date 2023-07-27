### import required packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,GRU, Dropout
import pickle
import warnings
warnings.filterwarnings('ignore')

### Code to create new dataset

## read data
# old_dataset = pd.read_csv('./data/q2_dataset.csv', parse_dates=['Date']) 
## sort data datewise
# old_dataset.iloc[:] = old_dataset.iloc[::-1].values 

## function to create new dataset
# def create_dataset(dataset, look_back):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back):
#         a = []
#         for j in range(i, i+look_back):
#             for k in range(2,6):
#                 a.append(dataset.iloc[j, k])
#         dataX.append(a)
#         dataY.append(dataset.iloc[i + look_back, 3])
#     return np.array(dataX), np.array(dataY)


# look_back = 3
# new_dataset_train, new_dataset_test = create_dataset(old_dataset, look_back)

## split data in train and test set
# X_train, X_test, y_train, y_test = train_test_split(new_dataset_train, new_dataset_test, test_size=0.30, random_state=42)

## save data into csv files
# train_data_RNN = pd.DataFrame(np.c_[X_train, y_train], columns=['Volume(t-3)','Open(t-3)','High(t-3)','Low(t-3)','Volume(t-2)','Open(t-2)','High(t-2)','Low(t-2)','Volume(t-1)','Open(t-1)','High(t-1)','Low(t-1)','Open(t)'])
# test_data_RNN = pd.DataFrame(np.c_[X_test, y_test], columns=['Volume(t-3)','Open(t-3)','High(t-3)','Low(t-3)','Volume(t-2)','Open(t-2)','High(t-2)','Low(t-2)','Volume(t-1)','Open(t-1)','High(t-1)','Low(t-1)','Open(t)'])
# train_data_RNN.to_csv('train_data_RNN.csv', index=False)
# test_data_RNN.to_csv('test_data_RNN.csv', index=False)

### Functions

## Preprocessing steps
def preprocessing_train(data):
	# Split data into X and y
	X_train = data.drop('Open(t)', axis = 1)
	y_train = data.loc[:,['Open(t)']]

	# Scale data
	scaler_x = MinMaxScaler(feature_range = (0,1))
	scaler_y = MinMaxScaler(feature_range = (0,1))

	# Fit the scaler using available training data
	input_scaler = scaler_x.fit(X_train)
	output_scaler = scaler_y.fit(y_train)

	# Apply the scaler to training data
	train_x_norm = input_scaler.transform(X_train)
	train_y_norm = output_scaler.transform(y_train)

	# Saved scalers to apply on test data
	pickle.dump(input_scaler, open("./data/input_scaler.pkl", 'wb'))
	pickle.dump(output_scaler, open("./data/output_scaler.pkl", 'wb'))

	# Convert input to 3D
	X_train_3d = train_x_norm.reshape(train_x_norm.shape[0],3,4)
	y_train_3d = train_y_norm

	return X_train_3d, y_train_3d


## Create and Train your network
def create_model_GRU(units, X_train_3d):
    model = Sequential()
    model.add(GRU(units = units, return_sequences = True, input_shape = [X_train_3d.shape[1], X_train_3d.shape[2]]))
    model.add(GRU(units = units,return_sequences=True))
    model.add(GRU(units = units,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae']) # Compile model
    return model

def fit_model(model, X_train_3d, y_train_3d):
    history = model.fit(X_train_3d, y_train_3d, epochs=200, validation_split=0.2, batch_size=32)
    return history

### main code
if __name__ == "__main__":
	data = pd.read_csv('./data/train_data_RNN.csv') # Load training data
	X_train_3d, y_train_3d = preprocessing_train(data) # Preprocess data

	model_gru = create_model_GRU(64, X_train_3d) # Create model

	print('Training and validation loss for each epoch :')
	history_gru = fit_model(model_gru, X_train_3d, y_train_3d) # Fit model
	print('Final training loss is : {:.4f}'.format(history_gru.history['loss'][-1]))
	model_gru.save("./models/Group_30_RNN_model") # Save model
