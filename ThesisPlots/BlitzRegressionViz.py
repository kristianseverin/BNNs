import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the numpy data
test_loss = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/test_loss_blitz.npy')
val_loss = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/val_loss_blitz.npy')

# read the accuracy data
accuracy = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_blitz.npy')
upper_ci = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/upper_ci_blitz.npy')
lower_ci = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/lower_ci_blitz.npy')

# read the prediction vs target data
prediction = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/preds_blitz.npy')
target = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/true_blitz.npy')

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

# plot the prediction vs target
plt.figure(figsize=(10, 6))
# density plot
sns.kdeplot(target.flatten(), label='Target')
sns.kdeplot(prediction.flatten(), label='Prediction')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()