import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

class TrainState(train_state.TrainState):
    """Extended train state with batch statistics for BatchNorm"""
    batch_stats: dict

def create_train_state(model, key, learning_rate):
    """Initialize model parameters and optimizer"""
    dummy_input = jnp.ones((1, 32, 32, 3))
    variables = model.init(key, dummy_input, training=True)
    params = variables['params']
    batch_stats = variables['batch_stats']

    optimizer = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats
    )

@jax.jit
def train_step(state, batch_images, batch_labels):
    """Single training step with gradient computation"""

    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_batch_stats = state.apply_fn(
            variables, batch_images, training=True, mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
        return loss, (logits, new_batch_stats)

    (loss, (logits, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats['batch_stats'])

    predictions = jnp.argmax(logits, axis=-1)
    true_labels = jnp.argmax(batch_labels, axis=-1)
    accuracy = jnp.mean(predictions == true_labels)

    return state, loss, accuracy

@jax.jit
def eval_step(state, batch_images, batch_labels):
    """Evaluation step without gradient computation"""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch_images, training=False)

    loss = optax.softmax_cross_entropy(logits, batch_labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    true_labels = jnp.argmax(batch_labels, axis=-1)
    accuracy = jnp.mean(predictions == true_labels)

    return loss, accuracy
