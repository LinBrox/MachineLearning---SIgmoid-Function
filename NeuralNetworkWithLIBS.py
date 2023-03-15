import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Read in the data from the arff file and convert it to a pandas dataframe
file = pd.read_csv('messidor_features.arff', header=None, comment='@')

# Define the header list for the dataframe
headerList = ['Quality Assessment', 'Pre-Screening', 'MA Detection 1', 'MA Detection 2', 'MA Detection 3',
              'MA Detection 4', 'MA Detection 5', 'MA Detection 6', 'Exudates Detection 1',
              'Exudates Detection 2', 'Exudates Detection 3', 'Exudates Detection 4',
              'Exudates Detection 5', 'Exudates Detection 6', 'Exudates Detection 7', 'Exudates Detection 8',
              'Euclidean Distance', 'OPTIC Disc', 'AM/FM', 'Output']
file.columns = headerList

# Create target variable and feature set
y = file['Output']
X = file.drop(['Output'], axis=1)

# Count the number of nulls in each column, excluding the first 2 and the last column
for col in file.iloc[:, 2:-1].columns:
    numberOfNulls = (file[col] == 0).sum()
    print("Number of Null Values", col, ":", numberOfNulls)

    # Drop columns that have more than 50% null values
    if numberOfNulls > 1151 * 0.25:
        file.drop(columns=[col], inplace=True)
    elif numberOfNulls != 0:
        # Calculate the average for the columns that have null values without using the 0 values
        avg = file[file[col] != 0][col]  # Get non-zero values
        avg_Val = avg.mean()  # Calculate the mean
        print("Average ", col, " (excluding 0 values):", avg_Val)

        # Replace all values equal to zero in the column with its average value
        file.loc[file[col] == 0, col] = avg_Val

# Normalise the feature set
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10000, batch_size=100, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print('Test accuracy:', test_acc)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate the confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred_binary)
print('Confusion matrix:')
print(cm)
print('Accuracy score:', accuracy_score(y_test, y_pred_binary))

# Plot the training and validation MSE vs epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE vs Epochs')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Calculate AUC score
test_auc = roc_auc_score(y_test, y_pred)
print('Test AUC:', test_auc)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
