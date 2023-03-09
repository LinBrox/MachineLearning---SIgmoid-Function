import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, X, y, hidden_neurons=20, lr=0.1, epochs=1000):
        self.X = X
        self.y = y
        self.hidden_neurons = hidden_neurons
        self.lr = lr
        self.epochs = epochs
        self.input_neurons = X.shape[1]
        self.output_neurons = y.shape[1]
        self.W1 = np.random.randn(self.input_neurons, self.hidden_neurons)
        self.W2 = np.random.randn(self.hidden_neurons, self.output_neurons)
        self.b1 = np.zeros((1, self.hidden_neurons))
        self.b2 = np.zeros((1, self.output_neurons))

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(self, x):
            return x * (1 - x)

        def softmax(self, x):
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        def forward(self, X):
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.softmax(self.z2)
            return self.a2

        def backward(self, X, y, y_pred):
            m = y.shape[0]
            dL_dz2 = y_pred - y
            dL_dW2 = (1 / m) * np.dot(self.a1.T, dL_dz2)
            dL_db2 = (1 / m) * np.sum(dL_dz2, axis=0, keepdims=True)
            dL_da1 = np.dot(dL_dz2, self.W2.T)
            dL_dz1 = dL_da1 * self.sigmoid_derivative(self.a1)
            dL_dW1 = (1 / m) * np.dot(X.T, dL_dz1)
            dL_db1 = (1 / m) * np.sum(dL_dz1, axis=0)

            self.W1 -= self.lr * dL_dW1
            self.b1 -= self.lr * dL_db1
            self.W2 -= self.lr * dL_dW2
            self.b2 -= self.lr * dL_db2

        def train(self):
            for i in range(self.epochs):
                y_pred = self.forward(self.X)
                self.backward(self.X, self.y, y_pred)
                loss = np.mean(-self.y * np.log(y_pred))
                if (i + 1) % 100 == 0:
                    print(f'Epoch: {i + 1}/{self.epochs} Loss: {loss:.4f}')

        def predict(self, X):
            y_pred = self.forward(X)
            predictions = np.argmax(y_pred, axis=1)
            return predictions


# Convert ARFF to CSV
file = pd.read_csv('messidor_features.arff', header=None, comment='@')
headerList = ['Quality Assessment', 'Pre-Screening', 'MA Detection 1', 'MA Detection 2', 'MA Detection 3',
              'MA Detection 4', 'MA Detection 5', 'MA Detection 6', 'Exudates Detection 1',
              'Exudates Detection 2', 'Exudates Detection 3', 'Exudates Detection 4',
              'Exudates Detection 5', 'Exudates Detection 6', 'Exudates Detection 7', 'Exudates Detection 8',
              'Euclidean Distance', 'OPTIC Disc', 'AM/FM', 'Output']
file.to_csv('updatedDF.csv', header=headerList, index=False)

# Read in the data from the csv & drop 1 column
df = pd.read_csv('updatedDF.csv')
df.drop(columns=['Output'], inplace=True)

for col in df.iloc[:, 2:-1].columns:
    numberOfNulls = (df[col] == 0).sum()
    print("Number of Null Values", col, ":", numberOfNulls)

    # Drop columns that have more than 50% null values
    if numberOfNulls > 1151 * 0.25:
        df.drop(columns=[col], inplace=True)
    elif numberOfNulls != 0:
        # Calculate the average for the columns that have null values without using the 0 values
        avg = df[df[col] != 0][col]  # Get non-zero values
        avg_Val = avg.mean()  # Calculate the mean
        print("Average ", col, " (excluding 0 values):", avg_Val)

        # Replace all values equal to zero in the column with its average value
        df.loc[df[col] == 0, col] = avg_Val

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int32)
# Normalize the features
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Build the neural network
X_shape = X_train.shape[1]
Y_shape = len(np.unique(y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(X_shape,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(Y_shape, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100000, batch_size=42, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Print the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print('Classification report:\n', classification_report(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from keras.callbacks import TensorBoard
#
# # Load and preprocess the dataset
# # Convert ARFF to CSV
# file = pd.read_csv('messidor_features.arff', header=None, comment='@')
# headerList = ['Quality Assessment', 'Pre-Screening', 'MA Detection 1', 'MA Detection 2', 'MA Detection 3',
#               'MA Detection 4', 'MA Detection 5', 'MA Detection 6', 'Exudates Detection 1',
#               'Exudates Detection 2', 'Exudates Detection 3', 'Exudates Detection 4',
#               'Exudates Detection 5', 'Exudates Detection 6', 'Exudates Detection 7', 'Exudates Detection 8',
#               'Euclidean Distance', 'OPTIC Disc', 'AM/FM', 'Output']
# file.to_csv('updatedDF.csv', header=headerList, index=False)
#
# # Read in the data from the csv
# df = pd.read_csv('updatedDF.csv')
#
# # Turns then to be a float value then drops the useless columns
# df.drop(columns=['Output'], inplace=True)
#
# # Define the number of times to train and test the model
# n = 5
#
# # Fill null values with the mean of each column
# for col in df.iloc[:, 2:-1].columns:
#     numberOfNulls = (df[col] == 0).sum()
#     print("Number of Null Values", col, ":", numberOfNulls)
#
#     # Drop columns that have more than 50% null values
#     if numberOfNulls > 1151 * 0.25:
#         df.drop(columns=[col], inplace=True)
#     elif numberOfNulls != 0:
#         # Calculate the average for the columns that have null values without using the 0 values
#         avg = df[df[col] != 0][col]  # Get non-zero values
#         avg_Val = avg.mean()  # Calculate the mean
#         print("Average ", col, " (excluding 0 values):", avg_Val)
#
#         # Replace all values equal to zero in the column with its average value
#         df.loc[df[col] == 0, col] = avg_Val
#
# X = df.iloc[:, :-1].values.astype(np.float32)
# y = file.iloc[:, -1].values.astype(np.int32)
# print(df)
#
# # Normalize the data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # Loop through training and testing the model n times
# for i in range(n):
#     print('Iteration', i + 1)
#
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Build the neural network
#     model = Sequential()
#     model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#
#     # Compile the model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     # Train the model
#     tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
#     model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[tensorboard_callback])
#
#     # Evaluate the model
#     _, accuracy = model.evaluate(X_test, y_test)
#     print('Accuracy: %.2f%%' % (accuracy * 100))
#     # Print the classification report and confusion matrix
#     from sklearn.metrics import classification_report, confusion_matrix
#
#     # Make predictions on the test set
#     y_pred = model.predict(X_test)
#     y_pred = np.argmax(y_pred, axis=1)
#
#     print('Classification report:\n', classification_report(y_test, y_pred))
#     print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
