import jax
import jax.numpy as jnp
from functools import partial


def get_s5_hiddens(input_sequence, ssm_layer):
    """Extract hidden states from S5 layer during forward pass.

    Args:
        input_sequence: Input sequence of shape (L, H)
        ssm_layer: Instance of S5SSM
    Returns:
        tuple: (outputs, hidden_states)
    """
    Lambda_elements = ssm_layer.Lambda_bar * jnp.ones((input_sequence.shape[0],
                                                       ssm_layer.Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: ssm_layer.B_bar @ u)(input_sequence)

    def binary_op(q_i, q_j):
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j * A_i, A_j * b_i + b_j

    _, hidden_states = jax.lax.associative_scan(binary_op,
                                                (Lambda_elements, Bu_elements))

    if ssm_layer.conj_sym:
        outputs = jax.vmap(lambda x: 2 * (ssm_layer.C_tilde @ x).real)(hidden_states)
    else:
        outputs = jax.vmap(lambda x: (ssm_layer.C_tilde @ x).real)(hidden_states)

    return outputs, hidden_states


def hidden_state_at_time(input_sequence, ssm_layer, time_idx, state_idx):
    """Get specific hidden state value.

    Args:
        input_sequence: Input sequence
        ssm_layer: S5SSM layer
        time_idx: Time step index
        state_idx: Hidden state dimension index
    Returns:
        Hidden state value at specified indices
    """
    _, states = get_s5_hiddens(input_sequence, ssm_layer)
    return jnp.sum(states[time_idx, state_idx])


def compute_hidden_gradients(input_batch, ssm_layer, time_indices, state_idx=0):
    """Compute gradients of hidden states w.r.t inputs.

    Args:
        input_batch: Batch of input sequences
        ssm_layer: S5SSM layer
        time_indices: Array of time indices to compute gradients for
        state_idx: Hidden state dimension to analyze (default 0)
    Returns:
        gradients: Array of gradient values
    """

    def grad_at_time(time_idx):
        grad_fn = jax.grad(hidden_state_at_time, argnums=0)
        grads = grad_fn(input_batch, ssm_layer, time_idx, state_idx)
        return jnp.abs(jnp.mean(grads, axis=0))

    return jax.vmap(grad_at_time)(time_indices)


def gradient_monitoring_hook(state, batch, time_indices, state_idx=0):
    """Hook function to monitor gradients during training.

    Args:
        state: Training state containing model
        batch: Current training batch
        time_indices: Time steps to analyze
        state_idx: Hidden state dimension to analyze
    Returns:
        mean_gradient: Mean gradient value across specified times
    """
    # get layer, set to 0 for now
    ssm_layer = state.model.ssm[0] if hasattr(state.model, 'ssm') else state.model.layers[0].ssm

    gradients = compute_hidden_gradients(batch['inputs'], ssm_layer, time_indices, state_idx)

    return jnp.mean(gradients)

