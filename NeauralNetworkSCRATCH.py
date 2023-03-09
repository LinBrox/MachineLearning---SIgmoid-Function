import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


class NeuralNetwork:
    def __init__(self, X, y, hidden_neurons=30, lr=0.1, epochs=100000):
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
        self.a2 = self.sigmoid(self.z2)
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
        # Initialize variables to keep track of MSE and number of epochs
        mse = []
        num_epochs = 0
        for i in range(self.epochs):
            # Calculate the MSE for this epoch
            y_pred = self.forward(self.X)
            epoch_mse = ((self.y - y_pred) ** 2).mean()
            mse.append(epoch_mse)
            # Increment the number of epochs
            num_epochs += 1
            self.backward(self.X, self.y, y_pred)

            loss = np.mean(-self.y * np.log(y_pred))
            if (i + 1) % 1 == 0:
                print(f'Epoch: {i + 1}/{self.epochs} Loss: {loss:.4f}')

        # Plot the training curve
        plt.plot(range(num_epochs), mse)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

    def predict(self, X):
        y_pred = self.forward(X)
        if y_pred.ndim == 1:
            return np.argmax(y_pred)
        else:
            return np.argmax(y_pred, axis=1)


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
y = df['Output']  # Create target variable
df.drop(columns=['Output'], inplace=True)

# Count the number of nulls in each column, excluding the first 2 and the last column
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

# Normalize the features
min_val = np.min(df)
max_val = np.max(df)
X_norm = (df - min_val) / (max_val - min_val)
print(X_norm)

# Determine number of rows to select
num_rows = int(len(X_norm) * 0.8)

# Select randomly 80% of rows
selected_data = X_norm.sample(n=num_rows, random_state=42)

# Drop selected rows from the original dataframe to obtain remaining 20%
remaining_data = X_norm.drop(selected_data.index)

print(selected_data)
print(remaining_data)

# Split data into training and test sets
X_train = selected_data.values
X_test = remaining_data.values
y_train = y.loc[selected_data.index].values
y_test = y.loc[remaining_data.index].values

# One-hot encode target variable
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Train neural network
nn = NeuralNetwork(X_train, y_train)
nn.train()

# Make predictions
y_pred = nn.predict(X_test)

# Print the shape of y_pred and y_test
print("Shape of y_pred:", y_pred.shape)
print("Shape of y_test:", y_test.shape)

# Convert one-hot encoded predictions and true labels back to class labels
# y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Construct the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate performance indices
fp_rate = fp / (fp + tn)
tp_rate = tp / (tp + fn)
accuracy = accuracy_score(y_test, y_pred)

# Print results
from tabulate import tabulate

headers = ['Metrics', 'Values']
# Define table rows
rows = [
    ['Confusion matrix', cm],
    ['FP rate', fp_rate],
    ['TP rate', tp_rate],
    ['Accuracy', accuracy]
]
# Print table
print(tabulate(rows, headers=headers, tablefmt='grid'))
# print('Confusion matrix:', cm)
# print('FP rate:', fp_rate)
# print('TP rate:', tp_rate)
# print('Accuracy:', accuracy)
