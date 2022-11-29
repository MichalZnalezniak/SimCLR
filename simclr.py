import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from torchvision import datasets, transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from torch.utils.data import Subset
import numpy
import pandas as pd
from hungarian import Hungarian, HungarianError, CoverZeros

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=f"./runs/{self.args.dataset_name}_TreeLevel{self.args.level_number}_lossAtAll{self.args.loss_at_all_level}_reg{self.args.regularization}reg_att_all{self.args.regularization_at_all_level}_pernode{self.args.per_node}_perlevel{self.args.per_level}")
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    
    def probability_vec_with_level(self, feature, level):
        prob_vec = torch.tensor([], device=self.args.device, requires_grad=True)
        for u in torch.arange(2**level-1, 2**(level+1) - 1, dtype=torch.long):
            probability_u = torch.ones_like(feature[:, 0], device=self.args.device, dtype=torch.float32)
            while(u > 0):
                if u/2 > torch.floor(u/2):
                    # Go left
                    u = torch.floor(u/2) 
                    u = u.long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # Go right
                    u = torch.floor(u/2) - 1
                    u = u.long()
                    probability_u *=  1 - feature[:, u]
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
        return prob_vec
    
    def probability_vec(self, feature):
        prob_vec = torch.tensor([], device=self.args.device, requires_grad=True)
        for u in torch.arange(2**self.args.level_number-1, 2**(self.args.level_number+1) - 1, dtype=torch.long):
            probability_u = torch.ones_like(feature[:, 0], device=self.args.device, dtype=torch.float32)
            while(u > 0):
                if u/2 > torch.floor(u/2):
                    # Go left
                    u = torch.floor(u/2) 
                    u = u.long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # Got right
                    u = torch.floor(u/2) - 1 
                    u = u.long()
                    probability_u *=  1 - feature[:, u]
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
        return prob_vec
            
            
    def binary_tree_loss(self, features, forward_mask):
        loss_value = torch.tensor([0], device=self.args.device, dtype=torch.float32)
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels * ~mask
        if self.args.loss_at_all_level:
            for level in range(1, self.args.level_number + 1):
                prob_features = self.probability_vec_with_level(features, level)
                prob_features = prob_features * forward_mask[level]
                # Calculate loss on positive classes
                # To avoid nan while calculating sqrt https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702  https://github.com/richzhang/PerceptualSimilarity/issues/69
                loss_value -= torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))
                # Calculate loss on negative classes
                loss_value += torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))
        else:
            prob_features = self.probability_vec(features)
            prob_features = prob_features * forward_mask[self.args.level_number]
            # Calculate loss on positive classes
            # To avoid nan while calculating sqrt https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702  https://github.com/richzhang/PerceptualSimilarity/issues/69
            loss_value -= torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))
            # Calculate loss on negative classes
            loss_value += torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))
     
        return loss_value
        

    def train(self, train_loader, valid_loader):
        print(self.model)
        torch.autograd.set_detect_anomaly(True)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        best_nmi = None
        self.masks_for_level = {level: torch.ones(2**level).cuda() for level in range(1, self.args.level_number+1)}
        for epoch_counter in range(self.args.epochs):
            tree_loss_array = []
            self.model.train()

            mean_of_probs_per_level_per_epoch = {level: torch.zeros(2**level).cuda() for level in range(1, self.args.level_number+1)}
            for i, (images, label) in enumerate(tqdm(train_loader)):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)
                self.optimizer.zero_grad()
                features = self.model(images)
                loss = self.binary_tree_loss(features, self.masks_for_level)
                tree_loss_array.append(loss)
                loss_sum = loss

                    
                if self.args.pruning:
                    for level in range(1, self.args.level_number+1):
                            prob_features = self.probability_vec_with_level(features, level)
                            probability_leaves = torch.mean(prob_features, dim=0)
                            probability_leaves_masked = self.masks_for_level[level] * probability_leaves
                            mean_of_probs_per_level_per_epoch[level] += probability_leaves_masked
                            probability_leaves_masked = probability_leaves_masked + 1e-8

                # loss to have unifrom dist on leaves
                if self.args.regularization:
                    loss_reg = torch.tensor([0], device=self.args.device, dtype=torch.float32)
                    if self.args.regularization_at_all_level:
                        for level in range(1, self.args.level_number+1):
                            prob_features = self.probability_vec_with_level(features, level)
                            probability_leaves = torch.mean(prob_features, dim=0)
                            probability_leaves_masked = self.masks_for_level[level] * probability_leaves
                            if self.args.per_level:
                                loss_reg += (-torch.sum((1/(2**level)) * torch.log(probability_leaves_masked)))
                            if self.args.per_node:
                                for leftnode in range(0,int((2**level)/2)):
                                    if not (self.masks_for_level[level][2*leftnode] == 0 or self.masks_for_level[level][2*leftnode+1] == 0):
                                        loss_reg -=   (1/(2**level)) * (0.5 * torch.log(probability_leaves_masked[2*leftnode]) + 0.5 * torch.log(probability_leaves_masked[2*leftnode+1]))
                    else:
                        loss_reg = torch.tensor([0], device=self.args.device, dtype=torch.float32)
                        if self.args.per_level:
                            loss_reg += (-torch.sum((1/(2**self.args.level_number)) * torch.log(probability_leaves_masked)))
                        if self.args.per_node:
                            for leftnode in range(0,int((2**self.args.level_number)/2)):
                                if not (self.masks_for_level[self.args.level_number][2*leftnode] == 0 or self.masks_for_level[self.args.level_number][2*leftnode+1] == 0):
                                    loss_reg -=  (0.5 * torch.log(probability_leaves_masked[2*leftnode]) + 0.5 * torch.log(probability_leaves_masked[2*leftnode+1]))
                    loss_sum += loss_reg
                
                loss_sum.backward()
                self.optimizer.step()
            print(f"Tree loss {sum(tree_loss_array) / len(tree_loss_array)}")
            self.writer.add_scalar('loss tree', loss, global_step=n_iter)
            logging.debug(f"Epoch: {epoch_counter}\tTree Loss: {loss}\t")

            with torch.no_grad():
                histograms_for_each_label_per_level = {self.args.level_number : numpy.array([numpy.zeros_like(torch.empty(2**self.args.level_number)) for i in range(0, 10)])}
                labels = []
                predictions = []
                self.model.eval()
                for i, (images, label) in enumerate(tqdm(valid_loader)):
                    images, label = images.cuda(), label.cuda()
                    feature = self.model(images) 
                    prob_features = self.probability_vec_with_level(feature, 4)
                    prob_features_masked = prob_features * self.masks_for_level[4]
                    for prediction, label in zip(torch.argmax(prob_features_masked.detach(), dim=1), label.detach()):
                        predictions.append(prediction.item())
                        labels.append(label.item())
                        histograms_for_each_label_per_level[self.args.level_number][label.item()][prediction.item()] += 1
                df_cm = pd.DataFrame(histograms_for_each_label_per_level[4], index = [class1 for class1 in range(0,10)], columns = [i for i in range(0,2**4)])
                tree_acc_val = tree_acc(df_cm)
                self.writer.add_scalar('tree_acc_val', tree_acc_val, epoch_counter)
                print(f'tree_acc_val {tree_acc_val} in epoch {epoch_counter}')
                logging.debug(f"tree_acc_val {tree_acc_val} in epoch {epoch_counter}\t")    

                actuall_nmi = normalized_mutual_info_score(labels, predictions)
                print(f'normalized_mutual_info_score_value {actuall_nmi} in epoch {epoch_counter}')
                logging.debug(f"normalized_mutual_info_score_value {actuall_nmi} in epoch {epoch_counter}\t")
                self.writer.add_scalar('normalized_mutual_info_score_value', actuall_nmi, epoch_counter)

                if best_nmi is None or actuall_nmi > best_nmi:
                    best_nmi = actuall_nmi
                    checkpoint_name = f'checkpoint_epoch_best_model.pth.tar'
                    save_checkpoint({
                        'epoch': self.args.epochs,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'mask': self.masks_for_level,
                        'nmi': best_nmi,
                        'tre_acc' :tree_acc_val
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))





                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss tree', loss, global_step=n_iter)
                    if self.args.regularization:
                        self.writer.add_scalar('loss reg', loss_reg, global_step=n_iter)               
                n_iter += 1
                
            if self.args.pruning:
                # if epoch_counter == 0:
                if epoch_counter >= self.args.start_pruning_epoch and epoch_counter < self.args.start_pruning_epoch + self.args.pruning_frequency*self.args.nodes_to_prune and epoch_counter % self.args.pruning_frequency == 0:
                    x = mean_of_probs_per_level_per_epoch[self.args.level_number]/i
                    x = x.double()
                    test = torch.where(x > 0.0, x, 1.0) 
                    self.masks_for_level[self.args.level_number][torch.argmin(test)] = 0
                    print(self.masks_for_level[self.args.level_number])
                
            if epoch_counter >= 10:
                self.scheduler.step()
            if self.args.regularization:
                logging.debug(f"Epoch: {epoch_counter}\t Regularization Loss: {loss_reg}\t")
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'mask': self.masks_for_level
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

def tree_acc(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    m = df.values.sum()
    df = df.values.tolist()
    hungarian = Hungarian()
    hungarian.calculate(df, is_profit_matrix=True)
    acc = 1.0*hungarian.get_total_potential()/m
    return acc