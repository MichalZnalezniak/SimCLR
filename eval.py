import io
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import torch.backends.cudnn as cudnn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import PIL
from torchvision import models
import argparse
import glob
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, CIFAROnlyKClasses
from torchvision.utils import save_image
from metrics import *
import os

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'mnist', 'svhn', 'fmnist'])
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
parser.add_argument('--save_point', default=".", type=str, help="Path to .pth ")
parser.add_argument('--regularization', default=False, action="store_true", help="Normalize to uniform")
parser.add_argument('--regularization_at_all_level', default=False, action="store_true", help="If regularization on all levels")
parser.add_argument("--loss_at_all_level", default=False, action="store_true",
                    help="if true calculate loss on all levels of tree")
                    
def eval():
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
   
    writer = SummaryWriter(log_dir=f"./eval/{args.save_point.split('/')[-1]}")

    # Load .pth
    model_file = glob.glob(args.save_point + "/*.pth.tar")
    print(model_file[0])
    checkpoint = torch.load(model_file[0], map_location=args.device)
    
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, args=args)
    print(model)
    model.load_state_dict(checkpoint['state_dict'])
    if 'mask' in checkpoint:
        masks_for_level = checkpoint['mask']
    print(masks_for_level)
    model = model.to(args.device)
    simclr = SimCLR(model=model, optimizer=None, scheduler=None, args=args)
    if args.dataset_name == 'cifar10':
        validset = datasets.CIFAR10('./', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
        ), download=True)     
        image_shape = torch.empty((3, 32, 32)) 
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset_name == 'mnist':
        validset = datasets.MNIST('./', train=False, transform=transforms.ToTensor(), download=True)
        image_shape = torch.empty((28, 28)) 
        classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9') 
    elif args.dataset_name == 'fmnist':
        validset = datasets.FashionMNIST('./', train=False, transform=transforms.ToTensor(), download=True)
        image_shape = torch.empty((28, 28)) 
        classes = ('T-shirt', 'Trouser', 'Pullover shirt','Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot',)
    elif args.dataset_name == 'svhn': 
        validset = datasets.SVHN('./', split='test', 
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                ]
        ), download=True)      
        image_shape = torch.empty((3, 32, 32)) 
        classes = ('1', '2', '3',
           '4', '5', '6', '7', '8', '9', '0')  
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    histograms_for_each_label_per_level = {level : np.array([np.zeros_like(torch.empty(2**level)) for i in range(0, 10)])  for level in range(1, args.level_number+1)}
    image_for_each_cluster_per_level = {level : np.array([np.zeros_like(image_shape) for i in range(0,2**args.level_number)])  for level in range(1, args.level_number+1)}
    images_in_leaves = {level : {key:0 for key in range(0,10)}  for level in range(0, 16)}

    model.eval()
    labels = []
    predictions = {level: [] for level in range(1, args.level_number + 1)}
    for i, (image, label) in enumerate(tqdm(valid_loader)):
        image, label = image.to(args.device), label.to(args.device)
        feature = model(image) 
        labels.append(label.detach().cpu().item())
        for level in range(1, args.level_number+1):
            prob_features = simclr.probability_vec_with_level(feature, level)
            prob_features_masked = prob_features * masks_for_level[level]
            histograms_for_each_label_per_level[level][label.item()][torch.argmax(prob_features_masked).item()] += 1
            image_for_each_cluster_per_level[level][torch.argmax(prob_features_masked).item()] += (image.squeeze().cpu().detach()).numpy()
            predictions[level].append(torch.argmax(prob_features_masked.detach().cpu()).unsqueeze(dim=0).item())
            if level == 4:
                if images_in_leaves[torch.argmax(prob_features_masked).item()][label.item()] < 20:
                    images_in_leaves[torch.argmax(prob_features_masked).item()][label.item()] += 1
                    os.makedirs(f"./{torch.argmax(prob_features_masked).item()}/{label.item()}", exist_ok=True)
                    # plt.imsave(f'./{torch.argmax(prob_features_masked).item()}/{label.item()}/image_{images_in_leaves[torch.argmax(prob_features_masked).item()][label.item()]}.png', numpy.transpose(image.squeeze(0), (1, 2, 0)))
                    save_image(image.squeeze(0), f'./{torch.argmax(prob_features_masked).item()}/{label.item()}/image_{images_in_leaves[torch.argmax(prob_features_masked).item()][label.item()]}.png')




    for level in range(1, args.level_number+1):
        df_cm = pd.DataFrame(histograms_for_each_label_per_level[level], index = [class1 for class1 in classes],
                    columns = [i for i in range(0,2**level)])
        plt.figure(figsize = (15,10))
        plt.title(f'Confusion matrix at level {level}')
        plt.xlabel('Cluster')
        plt.ylabel('Label')
        sn.heatmap(df_cm, annot=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)          
        writer.add_image(f'Confusion matrix at level {level}', image)
        for u in range(0,2**level):
            plt.figure(figsize = (10,7))
            plt.title(f'Mean of the images at level {level} that ended up in cluster number {u}')
            sum_per_label = sum([histograms_for_each_label_per_level[level][k][u] for k in range(0,10)])
            img = image_for_each_cluster_per_level[level][u] / sum_per_label
            if args.dataset_name == 'cifar10' or args.dataset_name == 'svhn':
                plt.imshow(np.transpose(img, (1, 2, 0)))
            else:
                plt.imshow(img, cmap='gray')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)          
            writer.add_image(f'Mean of the images at level {level} that ended up in cluster number {u}', image)
    plt.figure(figsize = (10,7))
    plt.hist(labels, bins=range(0,16), alpha=0.5, label="labels")
    plt.hist(predictions[args.level_number], bins=range(0,16), alpha=0.5, label="cluster prediction level 4")
    plt.xlabel("Data", size=14)
    plt.ylabel("Count", size=14)
    plt.legend(loc='upper right')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)          
    writer.add_image(f'Comparing histogram for cluster vs histogram for labels', image)
    for level in range(1, args.level_number+1): 
        df_cm = pd.DataFrame(histograms_for_each_label_per_level[level], index = [class1 for class1 in classes], columns = [i for i in range(0,2**level)])
        writer.add_scalar(f'adjusted_rand_score_at_{level}', adjusted_rand_score(labels, predictions[level]))
        writer.add_scalar(f'Leaf_Purity_at_{level}', LeafPurity(df_cm))
        writer.add_scalar(f'Average_Leaf_Purity_at_{level}', LeafPurity_mean(df_cm))
    writer.add_scalar('normalized_mutual_info_score_value', normalized_mutual_info_score(labels, predictions[args.level_number]))
    df_cm = pd.DataFrame(histograms_for_each_label_per_level[args.level_number], index = [class1 for class1 in classes], columns = [i for i in range(0,2**args.level_number)])
    dist_matrix = get_dist_matrix(df_cm, args.level_number)
    writer.add_scalar('Tree_accuracy', tree_acc(df_cm))
    writer.add_scalar('Dendrogram_Purity', dendrogram_purity(df_cm,args.level_number))
    plt.figure(figsize=(15, 15))
    plt.title(f'Class distance - heatmap')
    plt.xlabel('Class number')
    plt.ylabel('Class number')
    sn.color_palette("Spectral", as_cmap=True)
    sn.heatmap(dist_matrix, annot=True, vmin=0.0, vmax=args.level_number, fmt='.3f')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image(f'Class distance - heatmap', image)
    writer.close()
    
if __name__ == "__main__":
    eval()  
