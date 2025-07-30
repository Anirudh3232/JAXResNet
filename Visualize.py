import matplotlib.pyplot as plt
import numpy as np

# Extracted directly from your provided logs (30 epochs)
epochs = np.arange(1, 31)
train_losses = np.array([
    1.2582, 0.7763, 0.5596, 0.4330, 0.3273, 0.2451, 0.1829, 0.1297, 0.0968, 0.0827,
    0.0680, 0.0583, 0.0555, 0.0520, 0.0408, 0.0496, 0.0350, 0.0340, 0.0326, 0.0360,
    0.0319, 0.0316, 0.0251, 0.0257, 0.0276, 0.0195, 0.0323, 0.0206, 0.0212, 0.0243
])
train_accuracies = np.array([
    0.5507, 0.7250, 0.8049, 0.8490, 0.8854, 0.9137, 0.9354, 0.9545, 0.9668, 0.9702,
    0.9760, 0.9793, 0.9805, 0.9817, 0.9859, 0.9827, 0.9880, 0.9876, 0.9888, 0.9873,
    0.9886, 0.9895, 0.9915, 0.9908, 0.9908, 0.9930, 0.9890, 0.9929, 0.9923, 0.9918
])
test_losses = np.array([
    1.3868, 0.8688, 1.1708, 0.9100, 0.6547, 1.1166, 1.2518, 0.7445, 0.9087, 0.8998,
    0.7781, 0.7558, 0.9872, 0.8801, 1.1508, 0.9661, 0.7611, 0.9400, 1.1528, 0.9961,
    0.8847, 0.9116, 0.9472, 0.9847, 0.9452, 1.0498, 0.9509, 0.9840, 0.9759, 0.9747
])
test_accuracies = np.array([
    0.5639, 0.7189, 0.6425, 0.7323, 0.7956, 0.7284, 0.6923, 0.8059, 0.7866, 0.7928,
    0.8263, 0.8324, 0.8041, 0.8162, 0.7991, 0.8122, 0.8417, 0.8251, 0.8004, 0.8307,
    0.8316, 0.8295, 0.8252, 0.8235, 0.8292, 0.8185, 0.8321, 0.8274, 0.8315, 0.8296
])

# Plot loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(True)

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
