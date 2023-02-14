import torch
import argparse
import wandb

def save_checkpoint(state: dict, filename: str = './model_param/save.pt'):

    """Save model state and parameters"""
    torch.save(state, filename)

def parse_args():

    """
    Parse arguments given to the script.

    Returns
    -------
    argparse
        The parsed argument object
    """

    parser = argparse.ArgumentParser(
        description="Trainer for Hybrid Compression")
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs (single node)."
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to use WANDB, True/False"
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CERN",
        help="CERN, CERN2, CERN3, DRED, UMASS, UMASS2"
    )
    parser.add_argument(
        "--model",
        default='AE',
        type=str,
        choices=['AE', 'SCSAE', 'HAE', 'HAE2', 'HAE3'],
        help="model to be used"
    )
    parser.add_argument(
        "--batch",
        default=16,
        type=int,
        help="number of data samples in one batch"
    )
    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        help="number of total epochs to run"
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=lambda x: float(x),
        help="maximal learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0,
        type=lambda x: float(x),
        help="weight decay l2 regularization"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug mode by setting little datasets"
    )
    parser.add_argument(
        "--loss",
        default='MSE',
        type=str,
        choices=['MSE'],
        help="loss fn"
    )
    parser.add_argument(
        "--learning_rate_style",
        default='constant',
        type=str,
        choices=['constant', 'OneCycleLR'],
        help="learning rate scheduler (constant is default)"
    )
    parser.add_argument(
        "--optimiser",
        default='Adam',
        type=str,
        help="Adam"
    )
    parser.add_argument(
        "--latent_n",
        default=48,
        type=int,
        help="Dimension of latent vector"
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout of linear dimensions"
    )
    parser.add_argument(
        "--inputdays",
        default=12,
        type=int,
        help="Number of days (for AE, this is length * daily_freq, for HAE/CAE this is rows of dataframe)"
    )
    parser.add_argument(
        "--batchnorm",
        action="store_true",
        help="whether to use batch normalization for CNN AE"
    )
    parser.add_argument(
        "--noise",
        default='none',
        type=str,
        choices = ['none', 'gauss', 'speckle'],
        help="Type of noise to add"
    )
    parser.add_argument(
        "--noise_pct",
        default=0.1,
        type=float,
        help="Percentage of noise to be added"
    )
    parser.add_argument(
        "--dred_freq",
        default=60,
        type=int,
        help="The wide frequency for DRED dataset (1 second)"
    )

    args = parser.parse_args()
    return args

def start_wandb(args, checkpoint_path, project_name):
    
    """
    Initialise W&B for PyTorch experimentation tracking
    """

    wandb.login()

    config = dict(
        epochs=args.epochs,
        batch_size=args.batch,
        dataset=args.dataset,
        model=args.model,
        latent_dimension=args.latent_n,
        ae_input_days=args.inputdays,
        cnn_batchnorm=args.batchnorm,
        noise=args.noise,
        noise_pct=args.noise_pct,
        optimiser=args.optimiser,
        optimiser_hyperparams='default',
        learning_rate_style=args.learning_rate_style,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        checkpoint=checkpoint_path
    )

    run = wandb.init(project=project_name, entity=args.entity, config=config)

    return run