import argparse
from s5.utils.util import str2bool
from s5.test import test
from s5.dataloading import Datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument("--weights_path", type=str, default="./quick_test.pkl", help="path to weights file")
    parser.add_argument("--USE_WANDB", type=str2bool, default=False, help="log with wandb?")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name, e.g. username")
    parser.add_argument("--dir_name", type=str, default='./cache_dir', help="name of directory where data is cached")
    parser.add_argument("--dataset", type=str, choices=Datasets.keys(), default='mnist-classification', help="dataset name")

    # Model Parameters
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the network")
    parser.add_argument("--d_model", type=int, default=128, help="Number of features, i.e., H, dimension of layer inputs/outputs")
    parser.add_argument("--ssm_size", type=int, default=256, help="SSM size")
    parser.add_argument("--ssm_size_base", type=int, default=256, help="SSM latent size, i.e., P")
    parser.add_argument("--blocks", type=int, default=4, help="How many blocks, J, to initialize with")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--C_init", type=str, default="complex_normal", choices=["trunc_standard_normal", "lecun_normal", "complex_normal"], help="Options for initialization of C")
    parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"], help="Discretization method")
    parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last"], help="Pooling mode for classification tasks")
    parser.add_argument("--activation_fn", type=str, default="gelu", choices=["full_glu", "half_glu1", "half_glu2", "gelu"], help="Activation function")
    parser.add_argument("--conj_sym", type=str2bool, default=True, help="Whether to enforce conjugate symmetry")
    parser.add_argument("--clip_eigs", type=str2bool, default=True, help="Whether to enforce the left-half plane condition")
    parser.add_argument("--bidirectional", type=str2bool, default=False, help="Whether to use bidirectional model")
    parser.add_argument("--dt_min", type=float, default=0.001, help="Min value to sample initial timescale params from")
    parser.add_argument("--dt_max", type=float, default=0.1, help="Max value to sample initial timescale params from")

    # Optimization Parameters
    parser.add_argument("--prenorm", type=str2bool, default=True, help="True: use prenorm, False: use postnorm")
    parser.add_argument("--batchnorm", type=str2bool, default=False, help="True: use batchnorm, False: use layernorm")
    parser.add_argument("--bn_momentum", type=float, default=0.9, help="Batchnorm momentum")
    parser.add_argument("--bsz", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Max number of epochs")
    parser.add_argument("--early_stop_patience", type=int, default=1000, help="Number of epochs to continue training when val loss plateaus")
    parser.add_argument("--ssm_lr_base", type=float, default=1e-3, help="Initial SSM learning rate")
    parser.add_argument("--lr_factor", type=float, default=1, help="Global learning rate = lr_factor * ssm_lr_base")
    parser.add_argument("--dt_global", type=str2bool, default=False, help="Treat timescale parameter as global parameter or SSM parameter")
    parser.add_argument("--lr_min", type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--cosine_anneal", type=str2bool, default=True, help="Whether to use cosine annealing schedule")
    parser.add_argument("--warmup_end", type=int, default=1, help="Epoch to end linear warmup")
    parser.add_argument("--lr_patience", type=int, default=1000000, help="Patience before decaying learning rate for lr_decay_on_val_plateau")
    parser.add_argument("--reduce_factor", type=float, default=1.0, help="Factor to decay learning rate for lr_decay_on_val_plateau")
    parser.add_argument("--p_dropout", type=float, default=0.0, help="Probability of dropout")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay value")
    parser.add_argument("--opt_config", type=str, default="standard", choices=['standard', 'BandCdecay', 'BfastandCdecay', 'noBCdecay'], help="Optimization configuration")
    parser.add_argument("--jax_seed", type=int, default=0, help="Seed randomness")

    args = parser.parse_args()
    test(args)
