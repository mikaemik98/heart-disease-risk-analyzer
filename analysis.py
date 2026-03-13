import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Analysis module loaded successfully.")

columns = ['age', 'sex', 'cp', 'trestbps', 'cholesterol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

data = pd.read_csv('data/processed.switzerland.data', names=columns, na_values='?')

print(data.head())

data = data.dropna()

plt.scatter(data['age'], data['cholesterol'])

plt.xlabel('Age')
plt.ylabel('Cholesterol')

plt.title('Age vs Cholesterol')

plt.show()