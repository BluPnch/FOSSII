#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %pip install ucimlrepo
# %pip install pandas

# In[3]:


from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd

from datetime import datetime

# In[4]:


mushroom = fetch_ucirepo(id=73) 
  
X = mushroom.data.features 
y = mushroom.data.targets 
  
# print(X.head())
# print(X.shape)
# print(y.head())
# print(y.shape)


X = pd.get_dummies(X)

# print(X.head())
# print(X.shape)

y = y.iloc[:, 0].map({'e': 0, 'p': 1}).values

# print(y[:20])
# print(y.shape)
# print(np.unique(y))

# In[5]:


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=10):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_history = []

    def _step_function(self, x, threshold: int = 0):
        return np.where(x > threshold, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_history = []

        for _ in range(self.n_iterations):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self._step_function(linear_output)

                if y[i] != y_pred:
                    errors += 1

                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)

            self.errors_history.append(errors)

        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._step_function(linear_output)
        return y_pred



# In[6]:


indices = np.arange(len(X))

seed = int(datetime.now().timestamp())
np.random.seed(seed)
print("Использованный seed:", seed)

np.random.shuffle(indices)



TRAIN_DATA_PERCENT = 80
split_index = int(len(X) * TRAIN_DATA_PERCENT * 0.01)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X.values[train_indices]
y_train = y[train_indices]
X_test = X.values[test_indices]
y_test = y[test_indices]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# In[7]:


learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

# learning_rates = [0.01]


for lr in learning_rates:

    perceptron = Perceptron(
        learning_rate=lr,
        n_iterations=10
    )

    perceptron.fit(X_train, y_train)

    y_pred_test = perceptron.predict(X_test)

    accuracy = np.mean(y_pred_test == y_test)

    print(
        f"learning rate = {lr}, "
        f"accuracy = {accuracy:.3f}, "
        f"ошибок = {perceptron.errors_history[-1]}"
    )

# In[8]:


y_pred_train = perceptron.predict(X_train)
y_pred_test = perceptron.predict(X_test)

acc_train = np.mean(y_pred_train == y_train)
acc_test = np.mean(y_pred_test == y_test)

print(f"Accuracy (Train): {acc_train:.3f}")
print(f"Accuracy (Test): {acc_test:.3f}")

# In[9]:


import matplotlib.pyplot as plt

plt.plot(perceptron.errors_history)

plt.xlabel("Итерация")
plt.ylabel("Количество ошибок")
plt.title("Ошибки перцептрона во время обучения")

plt.grid()
plt.show()

# In[10]:


TP = 0
TN = 0
FP = 0
FN = 0

for real, predicted in zip(y_test, y_pred_test):

    if real == 1 and predicted == 1:
        TP += 1

    elif real == 0 and predicted == 0:
        TN += 1

    elif real == 0 and predicted == 1:
        FP += 1

    elif real == 1 and predicted == 0:
        FN += 1


confusion_matrix = np.array([
    [TN, FP],
    [FN, TP]
])
plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix)

plt.title("Confusion Matrix")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")

plt.xticks([0, 1], ["Съедобный", "Ядовитый"])
plt.yticks([0, 1], ["Съедобный", "Ядовитый"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix[i, j],
                 ha="center",
                 va="center")

plt.colorbar(label="Количество объектов")

plt.show()

# In[ ]:


accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) != 0
    else 0
)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")
