import torch
import torchvision
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import dataloader_usgs as dataloader
from PreResNet import *
import pdb
import numpy as np

BATCH_SIZE = 64     # controls how many images per visualization
NUM_BATCHES_TO_VISUALIZE = 10
TRAIN_DIR = "../USGS_data/crops/crops_square_32x32"
TEST_DIR = "../USGS_data/crops/crops_square_32x32_val"
NUM_CLASSES = 11
NET1_PATH = "checkpoint/net1.pth"
NET2_PATH = "checkpoint/net2.pth"
P_THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using', DEVICE)

def eval_train(model_name, model, all_loss, eval_loader, num_eval_samples, idx_to_class):   
    CE = nn.CrossEntropyLoss(reduction='none') 
    model.eval()
    losses = torch.zeros(num_eval_samples)
    all_targets = []
    all_features = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, features = model(inputs)   # outputs should be (64), features should be (64, 512)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
            all_targets.append(targets)
            all_features.append(features)
            if batch_idx % 100 == 0:
                print("batch", batch_idx)
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]       
      

    def plot_gmm_distribution(gmm, input_loss, output_fig_name)
        # plot probability distribution
        x_range = np.linspace(np.min(input_loss)-0.1, np.max(input_loss)+0.1, 200).reshape(-1, 1)
        GMM_curves = gmm.predict_proba(x_range)
        for i in range(GMM_curves.shape[1])
            labeled_curve = GMM_curves[:,i]
            fig, ax1 = plt.subplots()
                ax1.plot(x_range, labeled_curve, label=f"component {i}")
                ax1.set_xlabel("Loss")
                ax1.set_ylabel("Probability")
                ax1.legend()
                ax1.tick_params(axis='y')

        # plot loss histogram
        ax2 = ax1.twinx()
        ax2.hist(input_loss, alpha=0.5, bins=25)
        ax2.set_ylabel("Counts")
        ax2.tick_params(axis='y')
        
        fig.suptitle(f"{model_name}, component considered as clean: {gmm.means_.argmin()}")
        fig.tight_layout()
        plt.savefig(output_fig_name)
        plt.close()

    def plot_tsnes(all_targets, all_features, prob, idx_to_class, output_fig_name)
        # run tsne on feature points
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        print(model_name, "running tsne", all_targets.shape, all_features.shape)
        tsne_features = TSNE(n_components=2, perplexity=30, verbose=10).fit_transform(all_features)

        all_clean_indices = (prob > args.p_threshold).nonzero()[0]
        nrows = 3
        ncols = 4
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))
        # plt.figure(figsize=(16, 12))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', '#014d4e'] # 11 colors for 11 classes
        for class_number, class_name in idx_to_class.items():
            class_indices = (all_targets == class_number).nonzero()[0]
            class_clean_indices = np.intersect1d(all_clean_indices, class_indices)
            class_noisy_indices = np.setdiff1d(class_indices, class_clean_indices)
            class_clean_features = tsne_features[class_clean_indices, :]
            class_noisy_features = tsne_features[class_noisy_indices, :]

            # plt.scatter(class_clean_features[:, 0], class_clean_features[:, 1], marker=".", color=colors[class_number])
            # plt.scatter(class_noisy_features[:, 0], class_noisy_features[:, 1], marker="x", color=colors[class_number])
            # plt.plot([], [], label=class_name, color=colors[class_number])  # add label separately to cover both clean and noisy

            # plot clean/noisy on a separate plot, and also on the combined plot
            for axis in (ax[class_number // nrows, class_number % ncols], ax[nrows-1, ncols-1]):
                axis.scatter(class_clean_features[:, 0], class_clean_features[:, 1], marker=".", color=colors[class_number])
                axis.scatter(class_noisy_features[:, 0], class_noisy_features[:, 1], marker="x", color=colors[class_number])
                axis.plot([], [], label=class_name, color=colors[class_number])  # add label separately to cover both clean and noisy
            ax[class_number // nrows, class_number % ncols].set_title(class_name)
        ax[nrows-1, ncols-1].set_title(f"{model_name}, (n={all_targets.shape[0]}), X = noisy, O = clean")
        plt.savefig(output_fig_name)
        plt.close()

        # plt.legend()
        # plt.title(f"{model_name}, (n={all_targets.shape[0]}), X = noisy, O = clean")
        # plt.savefig(f"checkpoint/usgs_tsne_{model_name}.png")
        # plt.close()

    return prob,all_loss

loader = dataloader.usgs_dataloader(
    "usgs",
    r=0,
    noise_mode=None,
    batch_size=BATCH_SIZE,
    num_workers=1,\
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    log=[],
    noise_file=""
)  

# load pretrained models
net1 = ResNet18(num_classes=NUM_CLASSES)
net2 = ResNet18(num_classes=NUM_CLASSES)
net1_params = torch.load(NET1_PATH, map_location=torch.device('cpu'))
net2_params = torch.load(NET2_PATH, map_location=torch.device('cpu'))
net1.load_state_dict(net1_params)
net2.load_state_dict(net2_params)
net1.to(DEVICE)
net2.to(DEVICE)

print("models loaded")

eval_loader, num_eval_samples, idx_to_class = loader.run('eval_train')
print("running codivide for net1")
prob1, _ = eval_train("net1", net1,[],eval_loader,num_eval_samples, idx_to_class)   
print("running codivide for net2")
prob2, _ = eval_train("net2", net2,[],eval_loader,num_eval_samples, idx_to_class) 

# noisy1 = (prob1 < P_THRESHOLD)      
# noisy2 = (prob2 < P_THRESHOLD) 
# noisy = noisy1 & noisy2
# clean = ~noisy

# print("noisy images identified")
# unlabeled_trainloader = loader.run('inference', clean)

# # turn a list of BATCH_SIZE label strings into a single caption string with 8 captions/row
# def list_to_caption(labels):
#     caption = ""
#     for idx, label in enumerate(labels):
#         if idx % 8 == 0:
#             if idx != 0:
#                 caption += '\n'
#         else: 
#             caption += '|'
#         caption += label
#     return caption

# for idx, (img, target) in enumerate(unlabeled_trainloader):
#     grid = torchvision.utils.make_grid(img)

#     plt.figure(figsize=(14, 16))
#     plt.imshow(grid.permute(1, 2, 0))  # permute to (H, W, C) for matplotlib
#     plt.axis('off')
#     plt.suptitle(list_to_caption(target))
#     plt.savefig(f'checkpoint/noisy/noisy_sample_{idx}.png')
#     plt.clf()
#     plt.close()
# print("Done")