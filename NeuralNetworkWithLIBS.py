import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from NeauralNetworkSCRATCH import NeuralNetwork

# Load and preprocess the dataset
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Apply OneHotEncoder to the target variable
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train.values.reshape(-1, 1)).toarray()

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the neural network model
model = NeuralNetwork(X_train, y_train, hidden_neurons=30, lr=0.1, epochs=100000)
model.train()

# Check the shape of the one-hot-encoded array
print(y_train.shape)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Convert the one-hot-encoded predictions back to integer format
y_pred_int = onehotencoder.inverse_transform(y_pred.reshape(-1, 2))

# Reshape y_pred_int if necessary
if y_pred_int.ndim > 1 and y_pred_int.shape[1] > 1:
    y_pred_int = y_pred_int.argmax(axis=1)

# Convert the testing set target variable back to integer format
y_test_int = onehotencoder.inverse_transform(y_test)

# Print the accuracy score and confusion matrix
print('Accuracy score:', accuracy_score(y_test_int, y_pred_int))
print('Confusion matrix:\n', confusion_matrix(y_test_int, y_pred_int))
