import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Define activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


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

# Fill null values with the mean of each column
for col in df.columns:
    print("Number of zero values in column", col, ":", (df[col] == 0).sum())

    # Load dataset, visualize and drop uninformative columns
    df = pd.read_csv('messidor_features.csv')
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)
    df.drop(columns=['Exudates Detection 7', 'Exudates Detection 8', 'Output'], inplace=True)
    print(df)

    # 3. Normalize the features
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # 4. Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    # 5. Build the neural network
    input_shape = X_train.shape[1]
    hidden_shape = 20
    output_shape = len(np.unique(y_train))

    # Initialize weights with small random values
    w1 = np.random.randn(input_shape, hidden_shape) * 0.01
    w2 = np.random.randn(hidden_shape, output_shape) * 0.01

    # Set learning rate and number of iterations
    learning_rate = 0.01
    num_iterations = 1000

    # Set early stopping parameters
    patience = 10
    best_loss = np.inf
    best_w1 = None
    best_w2 = None
    no_improvement_count = 0

    # Train the model using backpropagation algorithm
    train_loss = []
    train_acc = []
    for i in range(num_iterations):
        # Forward propagation
        z1 = X_train.dot(w1)
        a1 = sigmoid(z1)
        z2 = a1.dot(w2)
        y_pred = sigmoid(z2)
        print(y_pred)

        # Calculate loss and accuracy
        loss = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
        acc = np.mean((y_pred > 0.5) == y_train)
        train_loss.append(loss)
        train_acc.append(acc)

        # Backward propagation
        dz2 = y_pred - y_train
        dw2 = a1.T.dot(dz2)
        dz1 = dz2.dot(w2.T) * sigmoid_derivative(z1)
        dw1 = X_train.T.dot(dz1)
        # Update weights
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2

        # Check early stopping condition
        if loss < best_loss:
            best_loss = loss
            best_w1 = w1.copy()
            best_w2 = w2.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping after {i + 1} iterations")
                w1 = best_w1
                w2 = best_w2
                break

        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{num_iterations}: loss={loss:.4f}, acc={acc:.4f}")

        # Test the model on the testing set
        z1 = X_test.dot(w1)
        a1 = sigmoid(z1)
        z2 = a1.dot(w2)
        y_pred = sigmoid(z2)

        # Convert y_pred to class labels
        y_pred = np.argmax(y_pred, axis=1)

        test_loss = -np.mean(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred))
        test_acc = np.mean(y_pred == y_test)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
