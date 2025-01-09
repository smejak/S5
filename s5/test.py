import pickle
from functools import partial

import jax
import jax.numpy as np
import optax
from flax.training import train_state
from jax import random
from jax.scipy.linalg import block_diag

from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel, RetrievalModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.train_helpers import map_nested_fn, validate


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


def test(args):
    # load dataset
    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 Testing on `{args.dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    ssm_init_fn = init_S5SSM(H=args.d_model,
                             P=ssm_size,
                             Lambda_re_init=Lambda.real,
                             Lambda_im_init=Lambda.imag,
                             V=V,
                             Vinv=Vinv,
                             C_init=args.C_init,
                             discretization=args.discretization,
                             dt_min=args.dt_min,
                             dt_max=args.dt_max,
                             conj_sym=args.conj_sym,
                             clip_eigs=args.clip_eigs,
                             bidirectional=args.bidirectional)
    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # load weights
    params = load_weights(args.weights_path)

    # initialize test state
    state = create_test_state(model_cls,
                              init_rng,
                              params,
                              padded,
                              retrieval,
                              in_dim=in_dim,
                              bsz=args.bsz,
                              seq_len=seq_len,
                              ssm_lr=ssm_lr,
                              lr=lr,
                              dt_global=args.dt_global)

    val_loss, val_acc = validate(state,
                                 model_cls,
                                 valloader,
                                 seq_len,
                                 in_dim,
                                 args.batchnorm)

    print(f"[*] Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    return val_loss, val_acc