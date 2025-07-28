import jax
import jax.numpy as jnp

SEED = 42

key = jax.random.PRNGKey(SEED)

NUM_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30

RESNET_BLOCKS = [2, 2, 2, 2]

INITIAL_CHANNELS = 64