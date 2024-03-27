import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the numpy data
test_loss = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/test_loss_blitz.np')
val_loss = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/val_loss_blitz.np')

# read the accuracy data
accuracy = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_blitz.np')
upper_ci = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/upper_ci_blitz.np')
lower_ci = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/lower_ci_blitz.np')

#plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(test_loss, label='Test Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(accuracy, label='Accuracy')
plt.plot(upper_ci, label='Upper CI')
plt.plot(lower_ci, label='Lower CI')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()