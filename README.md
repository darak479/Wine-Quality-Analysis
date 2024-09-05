Wine Quality Prediction:
This project explores a dataset of Portuguese wines to understand the factors that influence their quality. The goal is to analyze the chemical properties of the wines and build predictive models to classify them based on quality.

Dataset:
The dataset used in this project contains various physicochemical properties of the wines, including:

Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free Sulfur Dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol
Quality (score between 0 and 10)
Analysis
The analysis includes:

Exploratory Data Analysis (EDA) using histograms, scatter plots, and correlation matrices to visualize relationships between variables.
Data Preprocessing, including normalization using MinMaxScaler.
Model Building using Logistic Regression and Support Vector Machines (SVM) to predict wine quality.
Model Evaluation using metrics such as accuracy and confusion matrices.
How to Use
Clone the repository.
Install the required libraries (pandas, numpy, seaborn, matplotlib, scikit-learn).
Run the Jupyter Notebook wine-quality.ipynb to perform the analysis and train the models.
Results
The models achieved decent accuracy in predicting wine quality based on the chemical properties. The analysis provides insights into the factors that contribute to a good quality wine.

Future Work
Experiment with other machine learning algorithms to improve predictive performance.
Incorporate additional features or datasets to enhance the analysis.
Develop a web application to allow users to input wine properties and get a quality prediction.
Feel free to explore the code and contribute to this project! 🍷


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

# Load the wine quality dataset
data = pd.read_csv("/content/wine-quality.csv")

# Display the first few rows of the dataset
data.head()

# Show the column names
data.columns

# Get a summary of the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Generate descriptive statistics
data.describe().T

# Create histograms for each feature
data.hist(bins=20, figsize=(10,10))
plt.show()

# Count the occurrences of each quality rating
plt.figure(figsize=(6,4))
sb.countplot(x='quality', data=data,palette='Blues')
plt.show()

# Visualize the relationship between density and fixed acidity
plt.figure(figsize=(10,6))
sb.scatterplot(x='density', y='fixed acidity', data=data, palette='Reds')
plt.show()

# Explore the distribution of alcohol content across different quality ratings
plt.figure(figsize=(10,6))
sb.boxenplot(x='quality', y='alcohol', data=data)
plt.show() 

# Display a heatmap of correlations between features with correlation > 0.7
plt.figure(figsize=(10,10))
sb.heatmap(data.corr() > 0.7, annot=True, cbar=False)
plt.show() 

# Import necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Create a binary classification target variable ('best quality')
data['best quality'] = [1 if x > 5 else 0 for x in data.quality]

# Separate features and target variable
feature= data.drop(['quality', 'best quality'], axis=1)
target= data['best quality']

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(feature, target, test_size=0.2, random_state=40)
xtrain.shape, xtest.shape

# Normalize features using MinMaxScaler
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Train and evaluate Logistic Regression and SVM models
models = [LogisticRegression(), SVC(kernel='rbf')]

for i in range(2):
    models[i].fit(xtrain, ytrain)
    
    print(f'{models[i]} : ')
    print('Training Accuracy: ',metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy: ', metrics.roc_auc_score(ytest, models[i]. predict(xtest)))
    print() 

# Fit and predict using Logistic Regression
reg = LogisticRegression()
reg.fit(xtrain, ytrain)
regpred = reg.predict(xtest)

# Visualize confusion matrix for Logistic Regression
plt.figure(figsize=(6,4))
sb.heatmap(confusion_matrix(ytest, regpred), annot=True, cmap="RdPu")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show() 

# Display classification report and accuracy for SVM
print(metrics.classification_report(ytest, models[1].predict(xtest)))
print("Accuracy of SVM is : ", metrics.accuracy_score(ytest, models[1]. predict(xtest)))

# Count the occurrences of each original quality rating
data['quality'].value_counts()
