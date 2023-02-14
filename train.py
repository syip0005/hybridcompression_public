from pickle import FALSE
from lib.utils.dataloader import DRED_Dataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
import random
import os

import _init_paths

from models.models import AE, SCSAE, HAE, HAE_V2, HAE_V3
from utils.dataloader import CERN_Dataset, CERN_Dataset_V2, CERN_Dataset_V3, UMASS_Dataset, UMASS_Dataset_V2, DRED_Dataset
from utils.other import save_checkpoint, parse_args, start_wandb

# Force deterministic behaviour
SEED = 101 # TODO: move this to args
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global hyper parameters   
CACHE_DIR = './data'

# Generate unique number for checkpoint file
checkpoint_unique = False
while checkpoint_unique is False:
    CHECKPOINT_PATH = './model_param/state_' + str(random.randint(1000000, 9999999)) + '.pt'
    if os.path.exists(CHECKPOINT_PATH) is False:
        checkpoint_unique = True

random.seed(SEED)  # Comes after CHECKPOINT_PATH

def setup(args):
    """
    Sets up environment for multi-GPU processing
        Follows: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    # 

    os.environ['MASTER_ADDR'] = 'localhost'  # nb: M3 only supports single node multi-gpu (?)
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size) # nccl supports multi-GPU.
    torch.cuda.set_device(args.rank)

def cleanup():
    """
    Shuts down PyTorch group process
    """

    dist.destroy_process_group()

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler):

    """
    Training loop within a single epoch

    Returns
    ------
    float
        average of training losses for all batches
    """

    model.train()

    epoch_train_loss_avg = 0

    for input, input_noisy in tqdm.tqdm(train_loader):

        # Move all tensors to GPUs
        if args.gpus > 1:
            input = input.cuda(args.rank)
            input_noisy = input_noisy.cuda(args.rank)
        elif args.gpus == 1:
            input = input.cuda()
            input_noisy = input.cuda()

        # Typical training loop
        out, _ = model(input_noisy)

        loss = criterion(out, input)

        # Metrics
        with torch.no_grad():
            epoch_train_loss_avg += loss.item()

        # Gradient optimisation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None: # OneCycleLR is here, otherwise need to move out of loop.
            scheduler.step()

    with torch.no_grad():
        epoch_train_loss_avg /= len(train_loader)

    return epoch_train_loss_avg

def validate(args, val_loader, model, criterion, epoch):

    """
    Validation loop within a single epoch
    
    Returns
    ------
    float
        average of validation losses for all batches
    """

    model.eval()

    with torch.no_grad():

        epoch_val_loss_avg = 0

        for input, input_noisy in val_loader:

            # Move all tensors to GPU
            if args.gpus > 1:
                input = input.cuda(args.rank)
                input_noisy = input_noisy.cuda(args.rank)
            else:
                input = input.cuda()
                input_noisy = input.cuda()

            # Run model and calculate loss on val
            out, _ = model(input_noisy)
            loss = criterion(out, input)

            # Metrics
            epoch_val_loss_avg += loss.item()

        epoch_val_loss_avg /= len(val_loader)

    return epoch_val_loss_avg

def train_loop(args, model, device, train_loader, val_loader, sampler, run):

    """
    Training and validation loop

    Returns
    ------
    None
        Model is trained in place and logs are updated.
    """

    # Loss function
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    criterion.cuda(args.rank)

    # Initial learning rate
    if args.learning_rate_style == 'constant':
        initial_lr = args.lr
    elif args.learning_rate_style == 'OneCycleLR':
        initial_lr = args.lr * args.batch / 256 # TODO: check if this is correct
    else:
        raise NotImplementedError

    # Optimiser
    if args.optimiser == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               initial_lr, weight_decay=args.weight_decay)
    elif args.optimiser == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                               initial_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # Scheduler
    if args.learning_rate_style == 'constant':
        scheduler = None
    elif args.learning_rate_style == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                  steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.3)
    else:
        raise NotImplementedError

    # Use validation loss to checkpoint model state (assume initialises less than 1, i.e. smaller better)
    best_metric = 1

    for epoch in range(args.epochs):

        if args.gpus > 1:
            sampler.set_epoch(epoch)

        train_loss = train(args, train_loader, model, criterion, optimizer, epoch, scheduler)
        val_loss = validate(args, val_loader, model, criterion, epoch)

        if args.rank == 0 or args.rank is None: # Check if driver node, or only 1 GPU. 
            
            if args.wandb:
                run.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

            # Checkpoint
            if val_loss < best_metric:

                best_metric = val_loss

                if args.wandb:
                    run.summary["best_val_loss"] = best_metric

                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'best_metric': best_metric}, filename=CHECKPOINT_PATH)

                print("Best metric achieved.")

            print(
                f'Epoch: {epoch + 1} | Train Loss: {train_loss: .5f} | Val Loss: {val_loss: .5f}'
            )

def main(rank, args):

    """
    Main function to be spawned.
    """

    args.rank = rank

    if args.wandb and (args.rank == 0 or args.rank is None):
        RUN = start_wandb(args, CHECKPOINT_PATH, 'hybridcompression')
    else:
        RUN = None

    if args.gpus > 1:
        setup(args)
        args.batch = int(args.batch / args.gpus)
        print(args.batch)
    if args.debug:
        dataset_size_train = 100000
    else:
        dataset_size_train = None

    mode_dict = {'AE': ['fc', 2], 'SCSAE': ['cnn', 2], 'HAE': ['cnn', 1], 'HAE2': ['cnn', 1], 'HAE3': ['cnn', 1]} # TODO: refactor this into dictionary of dictionaries
    mode = mode_dict[args.model]

    # TODO: fix for dred later
    if args.dataset == 'DRED' and args.model == 'SCSAE':
        mode = ['cnn', 1]

    if args.dataset == 'CERN':
        train_iter = CERN_Dataset('./dataset/interim/', train=True, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays, N=dataset_size_train, noise_type=args.noise, noise_pct=args.noise_pct)
        val_iter = CERN_Dataset('./dataset/interim/', train=False, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays,  noise_type='none', noise_pct=args.noise_pct)
        WIDE_FREQ = 48
    elif args.dataset == 'CERN2':
        train_iter = CERN_Dataset_V2('./dataset/interim/', train=True, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays, N=dataset_size_train, noise_type=args.noise, noise_pct=args.noise_pct)
        val_iter = CERN_Dataset_V2('./dataset/interim/', train=False, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays,  noise_type='none', noise_pct=args.noise_pct)
        WIDE_FREQ = 48
    elif args.dataset == 'CERN3':
        train_iter = CERN_Dataset_V3('./dataset/interim/', train=True, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays, N=dataset_size_train, noise_type=args.noise, noise_pct=args.noise_pct)
        val_iter = CERN_Dataset_V3('./dataset/interim/', train=False, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays,  noise_type='none', noise_pct=args.noise_pct)
        WIDE_FREQ = 48
    elif args.dataset =='UMASS':
        train_iter = UMASS_Dataset('./dataset/interim/', train=True, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays, N=dataset_size_train, noise_type=args.noise, noise_pct=args.noise_pct)
        val_iter = UMASS_Dataset('./dataset/interim/', train=False, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays,  noise_type='none', noise_pct=args.noise_pct)
        WIDE_FREQ = 96
    elif args.dataset =='UMASS2':
        train_iter = UMASS_Dataset_V2('./dataset/interim/', train=True, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays, N=dataset_size_train, noise_type=args.noise, noise_pct=args.noise_pct)
        val_iter = UMASS_Dataset_V2('./dataset/interim/', train=False, mode=mode[0], reshape_factor=mode[1], no_days=args.inputdays,  noise_type='none', noise_pct=args.noise_pct)
        WIDE_FREQ = 96
    elif args.dataset =='DRED':
        train_iter = DRED_Dataset('./dataset/interim/', train=True, mode=mode[0], reshape_factor=mode[1], # TODO: must fix hard code here, refactor to list above...
                                    no_rows=args.inputdays, N=dataset_size_train, noise_type=args.noise, noise_pct=args.noise_pct,
                                    wide_freq = args.dred_freq)
        val_iter = DRED_Dataset('./dataset/interim/', train=False, mode=mode[0], reshape_factor=mode[1], 
                                    no_rows=args.inputdays,  noise_type='none', noise_pct=args.noise_pct,
                                    wide_freq = args.dred_freq)
        WIDE_FREQ = args.dred_freq
    else:
        raise NotImplementedError

    if args.gpus > 1:
        TRAIN_SAMPLER = torch.utils.data.distributed.DistributedSampler(train_iter, num_replicas=args.gpus,
                                                                        rank=args.rank, shuffle=False,
                                                                        drop_last=True)
    else:
        TRAIN_SAMPLER = None

    TRAIN_LOADER = DataLoader(train_iter, batch_size=args.batch, drop_last=False, shuffle=False,
                              sampler=TRAIN_SAMPLER)
    VAL_LOADER = DataLoader(val_iter, batch_size=args.batch, drop_last=False, shuffle=False)

    # Model init
    if args.model == 'AE':
        MODEL = AE(input_days = args.inputdays, latent_n = args.latent_n, dropout = args.dropout,
                        wide_freq = WIDE_FREQ)
    elif args.model == 'SCSAE':
        MODEL = SCSAE(batchnormalize = args.batchnorm, latent_n = args.latent_n, dropout = args.dropout,
                        input_days = args.inputdays, wide_freq = WIDE_FREQ, reshape_factor = mode[1])
    elif args.model == 'HAE':
        MODEL = HAE(batchnormalize = args.batchnorm, latent_n = args.latent_n, dropout = args.dropout,
                        input_days = args.inputdays, wide_freq = WIDE_FREQ, reshape_factor = mode[1])
    elif args.model == 'HAE2':
        MODEL = HAE_V2(batchnormalize = args.batchnorm, latent_n = args.latent_n, dropout = args.dropout,
                        input_days = args.inputdays, wide_freq = WIDE_FREQ, reshape_factor = mode[1], device = 'cuda') # TODO: fix in case we ever use cluster
    elif args.model == 'HAE3':
        MODEL = HAE_V3(batchnormalize = args.batchnorm, latent_n = args.latent_n, dropout = args.dropout,
                        input_days = args.inputdays, wide_freq = WIDE_FREQ, reshape_factor = mode[1], device = 'cuda') # TODO: fix in case we ever use cluster
    else:
        raise NotImplementedError

    if args.gpus > 1:
        MODEL.cuda(args.rank)
        MODEL = DDP(MODEL, device_ids=[args.rank], output_device=args.rank)
    elif args.gpus == 1:
        MODEL.cuda()

    if args.wandb and (args.rank == 0 or args.rank is None):
        RUN.watch(MODEL)

    train_loop(args, MODEL, DEVICE, TRAIN_LOADER, VAL_LOADER, TRAIN_SAMPLER, RUN)


if __name__ == "__main__":

    args = parse_args()

    args.world_size = args.gpus * 1  # As we are only using a single node

    assert torch.cuda.device_count() >= args.world_size, "GPU required less than available"

    if args.gpus > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    else:
        main(None, args)
