from __future__ import print_function
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import dataloader_usgs as dataloader
import numpy as np
import matplotlib.pyplot as plt
import collections

parser = argparse.ArgumentParser(description='PyTorch USGS Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--n_components_gmm', default=2, type=int, help='number of components in gmm')
parser.add_argument('--save_visualizations_interval', default=50, type=int)
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--r', default=0.2, type=float, help='noisy label rate')
parser.add_argument('--noise_mode', default='none', type=str, help='sym asym or none')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--conf_penalty', default=True, type=bool, help='whether to warm up with confidence penalty or not')
parser.add_argument('--balanced_softmax', action='store_true')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=456)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=11, type=int)
parser.add_argument('--train_data_path', default='../USGS_data/crops/crops_square_32x32', type=str, help='path to train dataset')
parser.add_argument('--test_data_path', default='../USGS_data/crops/crops_square_32x32_val', type=str, help='path to test dataset')
parser.add_argument('--dataset', default='usgs', type=str)
parser.add_argument('--pretrained_net1', default='', type=str, help='load the model from a pretrained .pth')
parser.add_argument('--pretrained_net2', default='', type=str, help='load the model from a pretrained .pth')
parser.add_argument('--output_dir', default='output', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(args)


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        Lx_record = []
        Lu_record = []

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11, _ = net(inputs_u)
            outputs_u12, _ = net(inputs_u2)
            outputs_u21, _ = net2(inputs_u)
            outputs_u22, _ = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()
            
            # label refinement of labeled samples
            outputs_x, _ = net(inputs_x)
            outputs_x2, _ = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits, _ = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = train_loss(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.warmup_epochs)
        Lx_record.append(Lx.item())
        Lu_record.append(Lu.item())

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('%s | Epoch [%3d/%3d]\t Mean labeled loss: %.2f  Mean unlabeled loss: %.2f'
            %(args.dataset, epoch, args.num_epochs, np.mean(Lx_record), np.mean(Lu_record)))

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    loss_record = []
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = warmup_loss(outputs, labels)
        if args.conf_penalty:   # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:   
            L = loss
        L.backward()
        optimizer.step() 
        loss_record.append(loss.item())

    print('%s | Epoch [%3d/%3d]\t Mean loss: %.4f'
            %(args.dataset, epoch, args.num_epochs, np.mean(loss_record)))

def plot_confusion_matrices(total_targets, total_predictions, classes, epoch):
    cm = confusion_matrix(total_targets, total_predictions, labels=range(args.num_class))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    cm_norm = confusion_matrix(total_targets, total_predictions, labels=range(args.num_class), normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)

    os.makedirs(f"{args.output_dir}/epoch_{epoch}", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.tick_params(axis='both', which='major', labelsize=13)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.savefig(f"{args.output_dir}/epoch_{epoch}/confusion_table.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.tick_params(axis='both', which='major', labelsize=13)
    disp_norm.plot(ax=ax, xticks_rotation='vertical')
    plt.savefig(f"{args.output_dir}/epoch_{epoch}/confusion_table_norm.png")
    plt.close()

def test(test_loader,epoch,net1,net2,idx_to_class):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    class_correct = collections.defaultdict(int)
    class_total = collections.defaultdict(int)
    total_predictions = []
    total_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1, _ = net1(inputs)
            outputs2, _ = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)

            total_predictions.append(predicted)
            total_targets.append(targets)
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item() 

            num_classes = outputs.shape[1]
            for c in range(num_classes):
                mask = (targets == c)
                class_total[c] += targets[mask].shape[0]
                class_correct[c] += predicted[mask].eq(targets[mask]).cpu().sum().item()

    acc = 100.*correct/total
    print(f"Test Epoch #{epoch}\t Accuracy: {acc}% = {correct}/{total}")  

    for c in class_total.keys():
        print(f"Class {idx_to_class[c]} correct: {class_correct[c]/class_total[c]} = {class_correct[c]}/{class_total[c]}")

    # relating to the confusion matrix
    total_predictions = torch.hstack(total_predictions).detach().cpu().numpy()
    total_targets = torch.hstack(total_targets).detach().cpu().numpy()
    classes = [idx_to_class[i] for i in range(args.num_class)]
    plot_confusion_matrices(total_predictions, total_targets, classes, epoch)


def plot_gmm_distribution(gmm, input_loss, output_fig_name):
    # plot probability distribution
    x_range = np.linspace(input_loss.min().item()-0.1, input_loss.max().item()+0.1, 200).reshape(-1, 1)
    GMM_curves = gmm.predict_proba(x_range)
    fig, ax1 = plt.subplots()
    for i in range(GMM_curves.shape[1]):
        labeled_curve = GMM_curves[:,i]
        ax1.plot(x_range, labeled_curve, label=f"component {i}")
        ax1.set_xlabel("Loss")
        ax1.set_ylabel("Probability")
        ax1.legend()
        ax1.tick_params(axis='y')

    # plot loss histogram
    ax2 = ax1.twinx()
    ax2.hist(input_loss.numpy().flatten(), alpha=0.5, bins=25)
    ax2.set_ylabel("Counts")
    ax2.tick_params(axis='y')
    
    fig.suptitle(f"GMM mixture, component considered as clean: {gmm.means_.argmin()}")
    fig.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()

def plot_tsnes(eval_dataset, all_targets, all_features, prob, epoch, model_name):
    # run tsne on feature points
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_features = torch.cat(all_features, dim=0).cpu().numpy()
    # tsne_features = all_features[:, :2]     # fake tsne to debug faster
    tsne_features = TSNE(n_components=2, perplexity=30).fit_transform(all_features)
    tsne_features = tsne_features.astype(np.float64)    # for json serialization

    idx_to_class = eval_dataset.get_idx_to_class_map()
    data_paths = eval_dataset.get_data_paths()

    all_clean_indices = (prob > args.p_threshold).nonzero()[0]
    nrows = 3
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(32, 24))
    combined_fig, combined_ax = plt.subplots(figsize=(16, 12))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', '#014d4e'] # 11 colors for 11 classes
    tsne_results = []
    for class_number, class_name in idx_to_class.items():
        class_indices = (all_targets == class_number).nonzero()[0]
        class_clean_indices = np.intersect1d(all_clean_indices, class_indices)
        class_noisy_indices = np.setdiff1d(class_indices, class_clean_indices)
        class_clean_features = tsne_features[class_clean_indices, :]
        class_noisy_features = tsne_features[class_noisy_indices, :]

        # plot clean/noisy on a separate plot, and also on the combined plot
        for axis in (ax[class_number // ncols, class_number % ncols], combined_ax):
            axis.scatter(class_clean_features[:, 0], class_clean_features[:, 1], marker=".", color=colors[class_number])
            axis.scatter(class_noisy_features[:, 0], class_noisy_features[:, 1], marker="x", color=colors[class_number])
            axis.plot([], [], label=class_name, color=colors[class_number])  # add label separately to cover both clean and noisy
        ax[class_number // ncols, class_number % ncols].set_title(class_name)

        # add this info to the json for interactive visualization
        for i in class_clean_indices:
            tsne_results.append({
                "class": class_name,
                "x": tsne_features[i, 0],
                "y": tsne_features[i, 1],
                "noisy": False,
                "filepath": data_paths[i]
            })
        for i in class_noisy_indices:
            tsne_results.append({
                "class": class_name,
                "x": tsne_features[i, 0],
                "y": tsne_features[i, 1],
                "noisy": True,
                "filepath": data_paths[i]
            })

    combined_ax.set_title(f"TSNE on ResNet18 before output layer, n={all_targets.shape[0]}, X = noisy, O = clean")
    combined_ax.legend()
    fig.savefig(f"{args.output_dir}/epoch_{epoch}/{model_name}_tsne.png")
    combined_fig.savefig(f"{args.output_dir}/epoch_{epoch}/{model_name}_tsne_combined.png")
    plt.close(fig)
    plt.close(combined_fig)

    with open(f"{args.output_dir}/epoch_{epoch}/{model_name}_tsne_data.json", "w") as f:
        tsne_dict = {}
        tsne_dict['data'] = tsne_results
        json.dump(tsne_dict, f)

def eval_train(model_name, model, all_loss, eval_dataset, eval_loader, epoch):    
    model.eval()
    losses = torch.zeros(len(eval_dataset))  
    all_targets = []
    all_features = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs, features = model(inputs) 
            all_targets.append(targets)
            all_features.append(features)

            loss = eval_loss(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
    #     history = torch.stack(all_loss)
    #     input_loss = history[-5:].mean(0)
    #     input_loss = input_loss.reshape(-1,1)
    # else:
    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=args.n_components_gmm,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()] 

    if epoch % args.save_visualizations_interval == 0 or epoch == args.warmup_epochs:
        os.makedirs(f"{args.output_dir}/epoch_{epoch}", exist_ok=True)
        print("Saving gmm visualizations, epoch:", epoch, "model:", model_name)
        plot_gmm_distribution(gmm, input_loss, f"{args.output_dir}/epoch_{epoch}/{model_name}_gmm.png")

    # if epoch == args.num_epochs:
        # os.makedirs(f"{args.output_dir}/epoch_{epoch}", exist_ok=True)
        # print("Saving final tsne visualizations, model:", model_name)
        # plot_tsnes(eval_dataset, all_targets, all_features, prob, epoch, model_name)

    return prob, all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1) + 1e-20   # for numerical stability
        loss = -torch.mean(torch.sum(probs.log()*probs, dim=1))
        # print("neg entropy loss", loss.item(), "| outputs max/min:", outputs.max().item(), outputs.min().item(), "| probs max/min:", probs.max().item(), probs.min().item())
        return loss

class BalancedSoftmax(object):
    def __init__(self, sample_per_class, reduction):
        """Init the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
                sample_per_class: A int tensor of size [no of classes].
                reduction: string. One of "none", "mean", "sum"
        """
        self.sample_per_class = sample_per_class
        self.reduction = reduction

    def __call__(self, logits, labels):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
                labels: A int tensor of size [batch].
                logits: A float tensor of size [batch, no_of_classes].
            Returns:
                loss: A float tensor. Balanced Softmax Loss.
        """
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels, reduction=self.reduction)
        return loss

def create_model(pretrained_path):
    model = ResNet18(num_classes=args.num_class)

    if pretrained_path != '':
        print('Loading pretrained model from', pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    loader = dataloader.usgs_dataloader(args.dataset,
        batch_size=args.batch_size,
        num_workers=2,\
        train_dir=args.train_data_path,
        test_dir=args.test_data_path,
        r=args.r,
        noise_mode=args.noise_mode,
    )

    print('Building models')
    net1 = create_model(args.pretrained_net1)
    net2 = create_model(args.pretrained_net2)
    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # these loaders don't need to be reset every time
    test_loader = loader.run('test') 
    warmup_trainloader = loader.run('warmup')
    eval_dataset, eval_loader = loader.run('eval_train')

    # calculate weights for weighted CE loss
    # use clean classes (bc noisy doesnt have as severe class imbalance)
    dict_train_counts_by_class_idx = loader.get_clean_class_counts_train()
    train_counts_by_class_idx = torch.zeros((len(dict_train_counts_by_class_idx)))
    for i, count in dict_train_counts_by_class_idx.items():
        train_counts_by_class_idx[i] = count

    print("Using balanced softmax?:", args.balanced_softmax)
    if args.balanced_softmax:
        warmup_loss = BalancedSoftmax(train_counts_by_class_idx, reduction='mean')
        eval_loss = BalancedSoftmax(train_counts_by_class_idx, reduction='none')
    else:
        warmup_loss = nn.CrossEntropyLoss(reduction='mean')
        eval_loss = nn.CrossEntropyLoss(reduction='none')   # bc we need the loss per-item
    train_loss= SemiLoss()
    if args.conf_penalty:
        conf_penalty = NegEntropy()
    all_loss = [[],[]] # save the history of losses from two networks

    for epoch in range(args.num_epochs+1):   
        lr=args.lr
        if epoch >= 150:
            lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr          
        
        if epoch<args.warmup_epochs: 
            print('Warmup Net1')
            warmup(epoch,net1,optimizer1,warmup_trainloader)    
            print('Warmup Net2')
            warmup(epoch,net2,optimizer2,warmup_trainloader) 
    
        else:         
            # first split the data 
            prob1,all_loss[0]=eval_train("net1",net1,all_loss[0],eval_dataset,eval_loader,epoch)   
            prob2,all_loss[1]=eval_train("net2",net2,all_loss[1],eval_dataset,eval_loader,epoch) 
                
            pred1 = (prob1 > args.p_threshold)      
            pred2 = (prob2 > args.p_threshold)
            print("Co-divide check model 1")
            loader.check_noisy(pred1)
            print("Co-divide check model 2")
            loader.check_noisy(pred2)
            
            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', transform='default', pred=pred2, prob=prob2) # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader) # train net1
            
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', transform='default', pred=pred1, prob=prob1) # co-divide
            train(epoch, net2, net1,optimizer2, labeled_trainloader, unlabeled_trainloader) # train net2        

        if epoch % 50 == 0 or epoch == args.warmup_epochs:
            test(test_loader,epoch,net1,net2,eval_dataset.get_idx_to_class_map())  

    print('Saving models')
    torch.save(net1.state_dict(), f'./{args.output_dir}/net1.pth')
    torch.save(net2.state_dict(), f'./{args.output_dir}/net2.pth')
