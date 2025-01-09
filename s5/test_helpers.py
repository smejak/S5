import pickle

import jax
import jax.numpy as np
import optax
from flax.training import train_state

from s5.train_helpers import map_nested_fn


def load_weights(pickle_file):
    with open(pickle_file, "rb") as f:
        params = pickle.load(f)
    return params


def create_test_state(model_cls,
                      rng,
                      params,
                      padded,
                      retrieval,
                      in_dim=1,
                      bsz=128,
                      seq_len=784,
                      ssm_lr=0,
                      lr=0,
                      dt_global=False):
    """
    Initializes the test state by loading provided parameters.

    :param model_cls:
    :param rng:
    :param params: The weights to be loaded in.
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param dt_global:
    :return:
    """

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2 * bsz, seq_len, in_dim)), np.ones(2 * bsz))
            integration_timesteps = np.ones((2 * bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))
        integration_timesteps = np.ones((bsz, seq_len,))

    model = model_cls(training=False)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng, "dropout": dropout_rng},
                           dummy_input, integration_timesteps)

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    print(f"[*] Loaded Parameters: {sum(jax.tree.leaves(param_sizes))}")

    """This option applies weight decay to C, but B is kept with the
        SSM parameters with no weight decay.
    """
    print("configuring standard optimization setup")
    if dt_global:
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "norm"]
            else ("none" if k in [] else "regular")
        )

    else:
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
            else ("none" if k in [] else "regular")
        )
    tx = optax.multi_transform(
        {
            "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
            "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                             weight_decay=0.01),
        },
        ssm_fn,
    )

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
