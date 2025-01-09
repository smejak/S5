from functools import partial

import jax.numpy as np
from jax import random
from jax.scipy.linalg import block_diag

from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel, RetrievalModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.test_helpers import load_weights, create_test_state
from s5.train_helpers import validate


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
