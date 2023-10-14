from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

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

# Create and fit the model
model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(features, target)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])
        id = int(request.form['id'])

        # Make a prediction using the model
        new_data = np.array([[age, sex, bmi, children, smoker, region, id]])
        predicted_class = model.predict(new_data)

        class_mapping = {0: 'Low',
                         1: 'Moderate', 
                         2: 'High', 
                         3: 'Very High'
                         }
        predicted_class_label = class_mapping[predicted_class[0]]

        # Calculate charges for the predicted class
        charges = df['charges'].values
        charges_mean = np.mean(charges)
        charges_std = np.std(charges)

        charge = (predicted_class[0] * charges_std) + charges_mean

        return render_template('result.html', predicted_class=predicted_class_label, predicted_charges=charge)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
