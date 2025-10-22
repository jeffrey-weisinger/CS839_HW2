import matplotlib.pyplot as plt

training_size = [100000, 3100000, 9100000, 21100000, 45100000]
kNN = [0.04534, 0.03924, 0.0386, 0.0600, 0.0330]
perplexity = [3.4218, 4.6387, 5.1553, 6.8097, 3.3397]

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(training_size, kNN, color="black", zorder=3, s=75)
plt.plot(training_size, kNN, linewidth=4)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("KNN Score")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(training_size, perplexity, color="black", zorder=3, s=75)
plt.plot(training_size, perplexity, linewidth=4)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Perplexity Score")
plt.show()

