```markdown
# Description

This chapter delves into SVM (Support Vector Machine) analysis using synthetic data from the study by Tang M. et al., 2021. The synthetic data used in this analysis contains three labels that are ordered, allowing for ordinal classification tasks. This Python code utilizes SVM to predict outcomes for synthetic data with three ordered labels. It includes data generation, model training with hyperparameter tuning using GridSearchCV, and evaluation on a test set.

## 1. Import Necessary Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

## 2. Create Synthetic Data

```python
# Set the random seed for reproducibility
np.random.seed(0)

# Means and standard deviations
means = np.array([[10, 10], [20, 10], [20, 20]])
stds = np.array([[5, 5], [5, 5], [7, 7]])
num_samples = 100

# Generate synthetic data and aggregate data and labels together
data_with_labels = []
for i, (mean, std) in enumerate(zip(means, stds)):
    cluster_data = np.random.normal(mean, std, size=(num_samples, 2))
    cluster_labels = np.full((num_samples, 1), i)
    data_with_labels.append(np.hstack((cluster_data, cluster_labels)))

data_with_labels = np.vstack(data_with_labels)
df = pd.DataFrame(data_with_labels, columns=['Variable 1', 'Variable 2', 'Label'])
df['Label'] = df['Label'] + 1

# Shuffle
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

# Plot the synthetic data
plt.figure(figsize=(8, 6))
for i in range(len(means)):
    plt.scatter(df[df['Label'] == i + 1]['Variable 1'], df[df['Label'] == i + 1]['Variable 2'], label=f'Cluster {i + 1}')

plt.title('Synthetic Data with Three Bivariate Gaussian Clusters')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.legend()
plt.grid(True)
plt.show()
```

The distribution of synthetic data is shown below:
![image](https://github.com/nanpapimolk/portfolio/assets/140955737/74707a8a-2dfc-45a1-aeea-63ac3addd897)

## 3. Data Preparation (Splitting Dataset)

We use a 70:30 split for the training and test sets.

```python
# Split the data into features (X) and labels (y)
X = df[['Variable 1', 'Variable 2']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

## 4. Hyperparameter Tuning and Model Training

We use SVM for the prediction model and tune hyperparameters with GridSearchCV.

```python
# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Initialize SVM classifier
svm = SVC(probability=True)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=5)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)
```

## 5. Model Evaluation

Evaluate the best model on the test set.

```python
# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
decision = best_model.decision_function(X_test)
prob_pred = best_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
```

The output of the evaluation is shown below:

![image](https://github.com/nanpapimolk/portfolio/assets/140955737/7000b290-b1a0-4842-8bef-4fd5c85b882e)

The model predicted with 73% accuracy.

### Reference

Tang, M., Perez-Fernandez, R., & De Baets, B. (2021). A comparative study of machine learning methods for ordinal classification with absolute and relative information. Knowledge-Based Systems, 230, 107358.
```
