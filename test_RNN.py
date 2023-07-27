## import required packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

def preprocessing_test(data_test):

	# Split train data to X and y
	X_test = data_test.drop('Open(t)', axis = 1)
	y_test = data_test.loc[:,['Open(t)']]

	input_scaler = pickle.load(open("./data/input_scaler.pkl", 'rb'))
	output_scaler = pickle.load(open("./data/output_scaler.pkl", 'rb'))

	# Apply the scaler to test data
	test_x_norm = input_scaler.transform(X_test)
	test_y_norm = output_scaler.transform(y_test)

	# Reshape the data
	X_test_3d = np.reshape(test_x_norm, (test_x_norm.shape[0],3,4))
	y_test_3d = test_y_norm

	# Inverse tranform
	y_test = output_scaler.inverse_transform(y_test_3d)
	X_test = X_test_3d	

	return X_test, y_test, output_scaler

def prediction(model, X_test, output_scaler):
    prediction = model.predict(X_test)
    prediction = output_scaler.inverse_transform(prediction)
    return prediction

# Evaluate predictions
def evaluate_prediction(predictions, actual):
	errors = predictions - actual
	mse = np.square(errors).mean()
	rmse = np.sqrt(mse)
	mae = np.abs(errors).mean()
	return mse, rmse, mae

# Plot true values vs predictions
def plot_future(prediction, y_test):
	plt.figure(figsize=(15,8))
	range = len(prediction)
	plt.plot(np.arange(range), np.array(y_test), label='True values', linestyle='dashed', color="red", marker='o')
	plt.plot(np.arange(range), np.array(prediction), label='Predictions', linestyle='dashed', color="blue", marker='o')
	plt.legend(loc='upper left')
	plt.xlabel('Time (day)')
	plt.ylabel('Open price')
	plt.show()

## Main function
if __name__ == "__main__":

	model_gru = keras.models.load_model("./models/Group_30_RNN_model") # Load your saved model
	 
	data_test = pd.read_csv('./data/test_data_RNN.csv') # Load testing data
	X_test, y_test, output_scaler = preprocessing_test(data_test)

	prediction_gru = prediction(model_gru, X_test, output_scaler)

	mse, rmse, mae = evaluate_prediction(prediction_gru, y_test) # MAE and RMSE
	print('Mean Absolute Error: {:.4f}'.format(mae))
	print('Root Mean Square Error: {:.4f}'.format(rmse))
	print('Test data loss is (Mean squared error is): {:.4f}'.format(mse))

	plot_future(prediction_gru, y_test) # Predictions vs actual values