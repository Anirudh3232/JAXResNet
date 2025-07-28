import jax
import jax.numpy as jnp

from config import SEED, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE        # Fixed - now specific imports
from data_utils import load_cifar10, preprocess_data, create_batches  # Fixed - now explicit functions
from resnet_model import create_model
from train_utils import create_train_state, train_step, eval_step

def main():
    # Correct: PRINGKey → PRNGKey; key/id names fixed
    key = jax.random.PRNGKey(SEED)
    key, init_key, data_key = jax.random.split(key, 3)

    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()

    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    print(f"Train set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")  # Fixed: correct shape print

    model = create_model()
    state = create_train_state(model, init_key, LEARNING_RATE)

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        # Training batches (shuffled)
        train_batch_images, train_batch_labels = create_batches(
            train_images, train_labels, BATCH_SIZE, data_key
        )

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_train_batches = train_batch_images.shape[0]  # Fixed: name

        for batch_idx in range(num_train_batches):
            state, loss, accuracy = train_step(
                state,
                train_batch_images[batch_idx],
                train_batch_labels[batch_idx]
            )
            epoch_loss += float(loss)
            epoch_accuracy += float(accuracy)

        epoch_loss /= num_train_batches
        epoch_accuracy /= num_train_batches

        # Evaluation phase
        test_batch_images, test_batch_labels = create_batches(
            test_images, test_labels, BATCH_SIZE, data_key
        )
        test_loss = 0.0
        test_accuracy = 0.0
        num_test_batches = test_batch_images.shape[0]

        for batch_idx in range(num_test_batches):
            loss, accuracy = eval_step(
                state,
                test_batch_images[batch_idx],
                test_batch_labels[batch_idx]
            )
            test_loss += float(loss)
            test_accuracy += float(accuracy)

        test_loss /= num_test_batches
        test_accuracy /= num_test_batches

        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss: {epoch_loss:.4f}, TrainAcc: {epoch_accuracy:.4f} | "
              f"Test loss: {test_loss:.4f}, TestAcc: {test_accuracy:.4f}")

        # For reproducibility across epochs (reshuffle batches)
        data_key = jax.random.split(data_key)[0]

    print("Training completed!")
    return state

if __name__ == "__main__":
    final_state = main()
