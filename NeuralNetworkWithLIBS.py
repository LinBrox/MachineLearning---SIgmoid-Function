import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

headerList = ['Quality Assessment', 'Pre-Screening', 'MA Detection 1', 'MA Detection 2', 'MA Detection 3',
              'MA Detection 4', 'MA Detection 5', 'MA Detection 6', 'Exudates Detection 1',
              'Exudates Detection 2', 'Exudates Detection 3', 'Exudates Detection 4',
              'Exudates Detection 5', 'Exudates Detection 6', 'Exudates Detection 7', 'Exudates Detection 8',
              'Euclidean Distance', 'OPTIC Disc', 'AM/FM', 'Output']

# Convert ARFF to CSV
with open('messidor_features.arff') as f:
    content = f.readlines()
content = [x.strip() for x in content]
start_data = content.index('@data') + 1
data = [x.split(',') for x in content[start_data:]]
df = pd.DataFrame(data, columns=headerList)
df.to_csv('messidor_features.csv', index=False)

# Read in the data from the csv
df = pd.read_csv('messidor_features.csv')

# Shows null values eventually will need to fill the remainder of the columns with Mean value
for col in df.columns:
    print("Number of zero values in column", col, ":", (df[col] == 0).sum())
# df.fillna(df.mean(), inplace=True)

# Turns then to be a float value then drops the useless columns
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int32)
df.drop(columns=['Exudates Detection 7', 'Exudates Detection 8', 'Output'], inplace=True)
print(df)

# Normalize the features
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Define the number of times to train and test the model
n = 10

# Loop through training and testing the model n times
for i in range(n):
    print('Iteration', i + 1)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    # Build the neural network
    X_shape = X_train.shape[1]
    Y_shape = len(np.unique(y_train))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_shape=(X_shape,), activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(Y_shape, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

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
