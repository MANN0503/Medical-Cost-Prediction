from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('train_data.csv')

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

# Create classes for classification based on 'charges'
df['class'] = pd.cut(df['charges'], bins=[0, 5000, 10000, 20000, np.inf], labels=['Low', 'Moderate', 'High', 'Very High'])

# Select features and target
features = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'id']]
target = df['class']

# Lists to store accuracy and model names
acc = []
model = []

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Decision tree classifier
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_value = DecisionTree.predict(Xtest)
accuracy = metrics.accuracy_score(Ytest, predicted_value)
acc.append(accuracy)
model.append('Decision Tree')
print("Decision Tree's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_value))

# Support Vector Machine (SVM) classifier
SVM = SVC(gamma='auto')
SVM.fit(Xtrain, Ytrain)
predicted_value = SVM.predict(Xtest)
accuracy = metrics.accuracy_score(Ytest, predicted_value)
acc.append(accuracy)
model.append('SVM')
print("SVM's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_value))

# Logistic Regression classifier
LogReg = LogisticRegression(random_state=2)
LogReg.fit(Xtrain, Ytrain)
predicted_value = LogReg.predict(Xtest)
accuracy = metrics.accuracy_score(Ytest, predicted_value)
acc.append(accuracy)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_value))

# Random Forest Classifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)
predicted_value = RF.predict(Xtest)
accuracy = metrics.accuracy_score(Ytest, predicted_value)
acc.append(accuracy)
model.append('Random Forest')
print("Random Forest's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_value))

# Accuracy Comparison
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='dark')

accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print(k, '-->', v)


new_data = np.array([[23, 1, 23.4, 0, 0, 1, 956]])


predicted_class = RF.predict(new_data)

class_mapping = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very High': 3}

predicted_class_numeric = [class_mapping[label] for label in predicted_class]

charges = df['charges'].values
charges_mean = np.mean(charges)
charges_std = np.std(charges)

predicted_charges = []
for class_label in predicted_class_numeric:
    charge = (class_label * charges_std) + charges_mean
    predicted_charges.append(charge)

for i in range(len(predicted_class)):
    print("Predicted Class:", predicted_class[i])
    print("Predicted Charges:", predicted_charges[i])
