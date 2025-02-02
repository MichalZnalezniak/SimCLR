import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import glob

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100', 'mnist', 'svhn' , 'fmnist', 'cifar10kclasses','imagenet10','imagenetdogs', 'imagenet30'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--level_number', default=3, type=int, help='Number of nodes of the binary tree')
parser.add_argument("--loss_at_all_level", default=False, action="store_true",
                    help="Flag to do something")
parser.add_argument("--gumbel", default=False, action="store_true",help="If gumbel sigmoid is used")
parser.add_argument("--temp", default=1.0,type=float,help='temp for gumbel softmax/sigmoid')
parser.add_argument('--save_point', default=".", type=str, help="Path to .pth ")
parser.add_argument('--load_model', default=False, action="store_true",help="Use pretrained model")
parser.add_argument('--regularization', default=False, action="store_true", help="Normalize to uniform")
parser.add_argument('--regularization_at_all_level', default=False, action="store_true", help="If regularization on all levels")
parser.add_argument('--weight', default=1.0, type=float)
parser.add_argument('--per_level', default=False, action="store_true", help="Normalize to uniform")
parser.add_argument('--per_node', default=False, action="store_true", help="Normalize to uniform")
parser.add_argument('--pruning', default=False, action='store_true', help="If true prune the leaves of the model")
parser.add_argument('--start_pruning_epoch',  default=0, type=int, help='Epoch when pruning starts')
parser.add_argument('--nodes_to_prune', default=6, type=int, help='Number of nodes to prune')
parser.add_argument('--pruning_frequency', default=60, type=int, help='Pruning frequency' )
parser.add_argument('--simclr_loss', default=True, action='store_true', help="If true train with SimCLR loss")


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    if args.dataset_name == 'imagenet10' or args.dataset_name == 'imagenetdogs' or args.dataset_name == 'imagenet30':
        if args.dataset_name == 'imagenet10':
            train_winds = [
                "n02056570",
                "n02085936",
                "n02128757",
                "n02690373",
                "n02692877",
                "n03095699",
                "n04254680",
                "n04285008",
                "n04467665",
                "n07747607"]
        elif args.dataset_name == 'imagenetdogs':
            train_winds = [
                "n02085936",
                "n02086646",
                "n02088238",
                "n02091467",
                "n02097130",
                "n02099601",
                "n02101388",
                "n02101556",
                "n02102177",
                "n02105056",
                "n02105412",
                "n02105855",
                "n02107142",
                "n02110958",
                "n02112137"
            ]
        elif args.dataset_name == 'imagenet30':
            train_winds = [
            "n12267677",
            "n02690373",
            "n02701002",
            "n01698640",
            "n02787622",
            "n02793495",
            "n02837789",
            "n03196217",
            "n02268443",
            "n03255030",
            "n03384352",
            "n03443371",
            "n03452741",
            "n07697537",
            "n03544143",
            "n03717622",
            "n03788195",
            "n03804744",
            "n03891332",
            "n03938244",
            "n04086273",
            "n03187595",
            "n04147183",
            "n04252077",
            "n04254680",
            "n01498041",
            "n07745940",
            "n04389033",
            "n04442312",
            "n09472597"]
        train_idx = [idx for idx, target in enumerate(train_dataset.wnids) if target in train_winds]
        train_indices = [idx for idx, target in enumerate(train_dataset.targets) if target in train_idx]
        train_dataset = Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, args=args)
    if args.load_model:
        model_file = glob.glob(args.save_point + "/*.pth.tar")
        print(f'Using Pretrained model {model_file[0]}')
        checkpoint = torch.load(model_file[0])
        model.load_state_dict(checkpoint['state_dict'])
        if 'mask' in checkpoint:
            model.masks_for_level = checkpoint['mask']



    print(model)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
