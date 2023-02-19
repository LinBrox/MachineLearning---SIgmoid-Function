import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

headerList = ['Quality Assessment', 'Pre-Screening', 'MA Detection 1', 'MA Detection 2', 'MA Detection 3',
              'MA Detection 4', 'MA Detection 5', 'MA Detection 6', 'Exudates Detection 1',
              'Exudates Detection 2', 'Exudates Detection 3', 'Exudates Detection 4',
              'Exudates Detection 5', 'Exudates Detection 6', 'Exudates Detection 7',
              'Euclidean Distance', 'OPTIC Disc', 'AM/FM', 'Output', 'Output2(Can be Dropped)']

# 1. Convert ARFF to CSV
with open('messidor_features.arff') as f:
    content = f.readlines()
content = [x.strip() for x in content]
start_data = content.index('@data') + 1
data = [x.split(',') for x in content[start_data:]]
df = pd.DataFrame(data, columns=headerList)
df.to_csv('messidor_features.csv', index=False)

# 2. Load dataset, visualize and drop uninformative columns
df = pd.read_csv('messidor_features.csv')
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int32)
plt.hist(X)
plt.show()
df.drop(columns=['Quality Assessment', 'OPTIC Disc', 'Output2(Can be Dropped)'], inplace=True)

# 3. Normalize the features
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# 4. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# 5. Build the neural network
import tensorflow as tf

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
