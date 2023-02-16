import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Task 1 - part 1
# Loads the ARFF and gives it the variable name to 'df' into the Pandas Dataframe
df = pd.read_csv('messidor_features.arff', header=None, comment='@')

# These are the headers used for the CSV file
headerList = ['Quality Assessment', 'Pre-Screening', 'MA Detection 1', 'MA Detection 2', 'MA Detection 3',
              'MA Detection 4', 'MA Detection 5', 'MA Detection 6', 'Exudates Detection 1',
              'Exudates Detection 2', 'Exudates Detection 3', 'Exudates Detection 4',
              'Exudates Detection 5', 'Exudates Detection 6', 'Exudates Detection 7',
              'Euclidean Distance', 'OPTIC Disc', 'AM/FM', 'Output', 'Output2(Can be Dropped)']

# Replace missing values with NaN
df.replace('?', np.nan, inplace=True)

# Print the number of null values before imputation
print("Number of null values before imputation:")
print(df.isnull().sum())

# Convert all columns to float type
df = df.astype(float)

# Impute missing values using KNN imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df))

# This line converts the arff that was in the pandas Dataframe 'df' into a CSV file
df_imputed.to_csv('file.csv', index=False, header=headerList)

# Print a few rows of the imputed dataframe
print("Imputed dataframe:")
print(df_imputed.head())

# Print the number of null values after imputation
print("Number of null values after imputation:")
print(df_imputed.isnull().sum())

# This line sets a variable to be able to read the document & then print it
pdf = pd.read_csv('file.csv')
print(pdf)

# END

# Task 1 - part 2 & 3
# Separate the input features and class labels
X = pdf.drop(['Output', 'Output2(Can be Dropped)'], axis=1)
y = pdf['Output2(Can be Dropped)']

# Normalize the input features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)


# Visualize the distribution of features and classes
# Histogram for visualization
pdf.drop(['Output', 'Output2(Can be Dropped)'], axis=1).hist(bins=20, figsize=(20, 15))
plt.show()

# Here is where I will drop the column(s) which I think do not contribute towards classification, for example
X_train = np.delete(X_train, [2], axis=1)
X_test = np.delete(X_test, [2], axis=1)

# END

