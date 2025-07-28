import tensorflow_datasets as tfds
import jax.numpy as jnp
import jax

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset"""
    train_ds, test_ds = tfds.load(
        'cifar10',
        split=['train', 'test'],
        as_supervised=True,
        batch_size=-1
    )

    train_images, train_labels = tfds.as_numpy(train_ds)
    test_images, test_labels = tfds.as_numpy(test_ds)

    return (train_images, train_labels), (test_images, test_labels)

def preprocess_data(images, labels):
    """Normalize images and convert labels to one-hot"""
    images = images.astype(jnp.float32) / 255.0
    images = 2.0 * images - 1.0
    labels = jax.nn.one_hot(labels, num_classes=10)
    return images, labels

def create_batches(images, labels, batch_size, key):
    """Create shuffled batches for training"""
    num_samples = images.shape[0]

    perm = jax.random.permutation(key, num_samples)
    images = images[perm]
    labels = labels[perm]

    num_batches = num_samples // batch_size

    batched_images = images[:num_batches * batch_size].reshape(
        num_batches, batch_size, *images.shape[1:]
    )
    batched_labels = labels[:num_batches * batch_size].reshape(
        num_batches, batch_size, *labels.shape[1:]
    )

    return batched_images, batched_labels
