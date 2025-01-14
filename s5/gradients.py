import jax
import jax.numpy as jnp
from functools import partial


def get_s5_hiddens(params, inputs, ssm_layer):
    """Extract hidden states from S5 layer during forward pass.

    Args:
        params: Layer parameters
        inputs: Input sequence of shape (L, H)
        ssm_layer: S5SSM layer class/function
    Returns:
        tuple: (outputs, hidden_states)
    """
    # Initialize the layer with parameters
    layer = ssm_layer.bind(params)

    Lambda_elements = layer.Lambda_bar * jnp.ones((inputs.shape[0],
                                                   layer.Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: layer.B_bar @ u)(inputs)

    def binary_op(q_i, q_j):
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j * A_i, A_j * b_i + b_j

    _, hidden_states = jax.lax.associative_scan(binary_op,
                                                (Lambda_elements, Bu_elements))

    # Get outputs using C_tilde
    if layer.conj_sym:
        outputs = jax.vmap(lambda x: 2 * (layer.C_tilde @ x).real)(hidden_states)
    else:
        outputs = jax.vmap(lambda x: (layer.C_tilde @ x).real)(hidden_states)

    return outputs, hidden_states


def hidden_state_at_time(params, inputs, ssm_layer, time_idx, state_idx):
    """Get specific hidden state value.

    Args:
        params: Layer parameters
        inputs: Input sequence
        ssm_layer: S5SSM layer class
        time_idx: Time step index
        state_idx: Hidden state dimension index
    Returns:
        Hidden state value at specified indices
    """
    _, states = get_s5_hiddens(params, inputs, ssm_layer)
    return jnp.sum(states[time_idx, state_idx])


def compute_hidden_gradients(params, input_batch, ssm_layer, time_indices, state_idx=0):
    """Compute gradients of hidden states w.r.t inputs.

    Args:
        params: Layer parameters
        input_batch: Batch of input sequences
        ssm_layer: S5SSM layer class
        time_indices: Array of time indices to compute gradients for
        state_idx: Hidden state dimension to analyze (default 0)
    Returns:
        gradients: Array of gradient values
    """

    def grad_at_time(time_idx):
        grad_fn = jax.grad(hidden_state_at_time, argnums=1)
        grads = grad_fn(params, input_batch, ssm_layer, time_idx, state_idx)
        return jnp.abs(jnp.mean(grads, axis=0))

    return jax.vmap(grad_at_time)(time_indices)


def gradient_monitoring_hook(state, batch, time_indices, state_idx=0):
    """Hook function to monitor gradients during training.

    Args:
        state: TrainState containing model parameters and state
        batch: Dictionary containing 'inputs' and 'labels'
        time_indices: Time steps to analyze
        state_idx: Hidden state dimension to analyze
    Returns:
        mean_gradient: Mean gradient value across specified times
    """
    # Extract SSM layer parameters (assuming first layer)
    ssm_params = state.params['ssm'][0] if 'ssm' in state.params else state.params['layers'][0]['ssm']
    ssm_layer = state.model.ssm[0] if hasattr(state.model, 'ssm') else state.model.layers[0].ssm

    # Compute gradients
    gradients = compute_hidden_gradients(
        params=ssm_params,
        input_batch=batch['inputs'],
        ssm_layer=ssm_layer,
        time_indices=time_indices,
        state_idx=state_idx
    )

    return jnp.mean(gradients)


# Make gradient computation JIT-compatible
gradient_monitoring_hook = jax.jit(gradient_monitoring_hook, static_argnames=['time_indices', 'state_idx'])