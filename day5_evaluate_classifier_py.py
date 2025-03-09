# -*- coding: utf-8 -*-
"""day5_evaluate_classifier.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18FMOWEkJnqiWFKApzu_hpjn7RdZrvaW3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

fs = 1000
t = np.arange(0, 1, 1/fs)
n_trials = 30000

def generate_trial(state):
    theta = np.random.uniform(0.3, 0.5) * np.sin(2 * np.pi * 5 * t)  # Random theta
    alpha = (np.random.uniform(0.6, 0.9) if state == "rest" else
             np.random.uniform(0.4, 0.6) if state == "focus" else
             np.random.uniform(0.1, 0.3)) * np.sin(2 * np.pi * 10 * t)
    beta = (np.random.uniform(0.1, 0.3) if state == "rest" else
            np.random.uniform(0.3, 0.5) if state == "focus" else
            np.random.uniform(0.6, 0.9)) * np.sin(2 * np.pi * 20 * t)
    noise = 2.8 * np.random.normal(0, 1, len(t))  # Bigger noise
    return theta + alpha + beta + noise

trials = np.array([generate_trial("rest") for _ in range(n_trials//3)] + [generate_trial("focus") for _ in range(n_trials//3)] + [generate_trial("move") for _ in range(n_trials//3)])
labels = np.array([0] * (n_trials//3) + [1] * (n_trials//3) + [2] * (n_trials//3))

# Added theta power to separate Focus/Move

features = []
for trial in trials:
  b_theta, a_theta = signal.butter(2, [4/(fs/2), 8/(fs/2)], btype='band')
  b_alpha, a_alpha = signal.butter(2, [8/(fs/2), 13/(fs/2)], btype='band')
  b_beta, a_beta =signal.butter(2, [13/(fs/2), 30/(fs/2)], btype='band')
  theta_signal = signal.filtfilt(b_theta, a_theta, trial)
  alpha_signal = signal.filtfilt(b_alpha, a_alpha, trial)
  beta_signal = signal.filtfilt(b_beta, a_beta, trial)
  theta_power = np.mean(theta_signal ** 2)
  alpha_power = np.mean(alpha_signal ** 2)
  beta_power = np.mean(beta_signal ** 2)
  features.append([theta_power, alpha_power, beta_power])
features = np.array(features)

clf_svm = SVC(kernel='rbf', random_state=42)
scores_svm = cross_val_score(clf_svm, features, labels, cv=5)
print(f"SVM CV Scores: {scores_svm}")
print(f"SVM Mean Accuracy: {scores_svm.mean():.2f} ± {scores_svm.std():.2f} ")

clf_lr = LogisticRegression(random_state=42)
scores_lr = cross_val_score(clf_lr, features, labels, cv=5)
print(f"Logistic CV Scores: {scores_lr}")
print(f"Logistic Mean Accuracy: {scores_lr.mean():.2f} ± {scores_lr.std():.2f}")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(cm_svm, display_labels=["Rest", "Focus", "Move"])
disp_svm.plot()
plt.title("SVM Confusion Matrix")
plt.show()

X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.3, random_state=42)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(cm_lr, display_labels=["Rest", "Focus", "Move"])
disp_lr.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

f1_per_class_svm = classification_report(y_test, y_pred_svm, output_dict=True)['macro avg']['f1-score']
f1_per_class_lr = classification_report(y_test, y_pred_lr, output_dict=True)['macro avg']['f1-score']
print(f"SVM F1 Score: {f1_per_class_svm}")
print(f"Logistic F1 Score: {f1_per_class_lr}")

print("SVM Classification Report:")
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print(classification_report(y_test, y_pred_svm, target_names=["Rest", "Focus", "Move"]))
print("Logistic Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Rest", "Focus", "Move"]))