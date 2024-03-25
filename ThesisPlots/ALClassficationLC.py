import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
accuracy_curves = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_curves.npy')
accuracy_seed = np.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_curve_seed.npy')

# plot the accuracy curves
plt.figure(figsize=(10,6))
plt.plot(accuracy_seed, label='Seed Model')
plt.plot(accuracy_curves[0], label='1 round Model')
plt.plot(accuracy_curves[1], label='2 rounds model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()

# a seaborn plot of the accuracy curves
sns.set_theme()
plt.figure(figsize=(10,6))
sns.lineplot(data=accuracy_seed)
sns.lineplot(data=accuracy_curves[0])
sns.lineplot(data=accuracy_curves[1])
plt.legend(['Seed Model', '1 round Model', '2 rounds Model'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.show()

