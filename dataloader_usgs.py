from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import torchvision
from torchnet.meter import AUCMeter
from collections import Counter
from copy import copy

# USGS
# mean=(0.23795913638363578, 0.3057805578709034, 0.3276135998876962)
# std=(0.2634323090563423, 0.22283197611992545, 0.21451868944431324)

class usgs_dataset(Dataset):
    def __init__(self, dataset, full_dataset, transform, mode, pred=[], probability=[]): 
        
        self.transform = transform
        self.mode = mode
        self.full_dataset = full_dataset

        # print("mode", self.mode, "root dir", root_dir)
        if self.mode=='test':
            self.data = full_dataset                  
        else:
            if self.mode == 'all':
                self.data = full_dataset
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    # not sure if this still works
                    # clean = (np.array(noise_label)==np.array(train_label))                                                       
                    # auc_meter = AUCMeter()
                    # auc_meter.reset()
                    # auc_meter.add(probability,clean)        
                    # auc,_,_ = auc_meter.value()               
                    # log.write('Number of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    # log.flush()      
                    
                elif self.mode == "unlabeled" or self.mode == "inference":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.data = torch.utils.data.Subset(full_dataset, pred_idx)                      
        class_counts = self.get_dataset_class_counts()
        print(self.mode, "has a size of", len(self.data), json.dumps(class_counts, indent=4)) 
                
    def __getitem__(self, index):
        if self.mode=='labeled': 
            assert index >= 0 and index < len(self.data)
            img, target = self.data[index]
            prob = self.probability[index]   
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img, _ = self.data[index]
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode=='all':
            img, target = self.data[index]
            img = self.transform(img)
            return img, target, index
        elif self.mode=='test':
            img, target = self.data[index]
            img = self.transform(img)
            return img, target
           
    def __len__(self):
        return len(self.data)   

    def get_idx_to_class_map(self):
        return {idx:_class for _class, idx in self.full_dataset.class_to_idx.items()}

    def get_dataset_class_counts(self):
        idx_to_class = self.get_idx_to_class_map()
        if isinstance(self.data, torch.utils.data.Subset):
            data_idx = self.data.indices
            targets = [self.data.dataset.targets[i] for i in data_idx]
            counts = Counter(targets)
        else: 
            targets = self.data.targets
            counts = Counter(self.data.targets)
        name_class_counts = {idx_to_class[idx]:count for idx, count in counts.items()}
        return name_class_counts

    def get_data_paths(self):
        if isinstance(self.data, torch.utils.data.Subset):
            full_dataset = self.data.dataset
            paths = [full_dataset.imgs[i][0] for i in self.data.indices]    # each element in ImageFolder.imgs is a tuple of (path, class)
        else:
            paths = [i[0] for i in self.data.imgs]
        return [os.path.join(*p.split('/')[-2:]) for p in paths]     # only interested in /Algae/000.jpg, not full data path

        
class usgs_dataloader():  
    def __init__(self, dataset, batch_size, num_workers, train_dir, test_dir, r, noise_mode):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.full_train_data = torchvision.datasets.ImageFolder(root=train_dir)
        self.full_test_data = torchvision.datasets.ImageFolder(root=test_dir)   

        # add noise to train data here, test data is untouched
        self.orig_train_labels = copy(self.full_train_data.targets)
        train_dataset_len = len(self.orig_train_labels)
        idx = list(range(train_dataset_len))
        random.shuffle(idx)
        self.noise_idx = idx[:int(r*train_dataset_len)]
        self.clean_idx = idx[int(r*train_dataset_len):]
        if noise_mode == 'sym':
            print("Adding symmetric noise with prob", r)
            for i in range(train_dataset_len):
                if i in self.noise_idx:
                    noiselabel = random.randint(0, len(self.full_train_data.classes)-1)
                    self.full_train_data.targets[i] = noiselabel
                    self.full_train_data.samples[i] = (self.full_train_data.samples[i][0], noiselabel)
        elif noise_mode == 'asym':
            # in order: ['Algae', 'Artificial', 'Bird', 'Bony Fish', 'Cartilaginous Fish', 'Debris', 'Glare', 'Invertebrate', 'Mammal', 'Reptile', 'Unknown_Other']
            # algae <--> unknown/other
            # artificial <--> debris
            # bird <--> reptile
            # bony fish <--> cartilaginous fish
            transition = {0:10, 1:5, 2:9, 3:4, 4:3, 5:1, 6:6, 7:7, 8:8, 9:2, 10:0}

            for i in range(train_dataset_len):
                if i in self.noise_idx:
                    noiselabel = transition[self.full_train_data.targets[i]]
                    self.full_train_data.targets[i] = noiselabel
                    self.full_train_data.samples[i] = (self.full_train_data.samples[i][0], noiselabel)

        self.transform_train_def = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.23795913638363578, 0.3057805578709034, 0.3276135998876962),
                    (0.2634323090563423, 0.22283197611992545, 0.21451868944431324)
                ),
            ]) 
        self.transform_train_aug = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2)),
                # transforms.ColorJitter(), 
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.23795913638363578, 0.3057805578709034, 0.3276135998876962),
                    (0.2634323090563423, 0.22283197611992545, 0.21451868944431324)
                ),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.23795913638363578, 0.3057805578709034, 0.3276135998876962),
                    (0.2634323090563423, 0.22283197611992545, 0.21451868944431324)
                ),
            ])

    def run(self,mode,transform='default',pred=[],prob=[]):
        if transform == 'default':
            transform_train = self.transform_train_def
            transform_test = self.transform_test
        elif transform == 'augmented':
            transform_train = self.transform_train_aug
            transform_test = self.transform_test

        if mode=='warmup':
            all_dataset = usgs_dataset(dataset=self.dataset, 
                full_dataset=self.full_train_data,
                transform=transform_train, 
                mode="all")
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = usgs_dataset(dataset=self.dataset, 
                full_dataset=self.full_train_data,
                transform=transform_train,
                mode="labeled", 
                pred=pred, 
                probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = usgs_dataset(dataset=self.dataset, 
                full_dataset=self.full_train_data,
                transform=transform_train, 
                mode="unlabeled", 
                pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = usgs_dataset(dataset=self.dataset, 
                full_dataset=self.full_test_data,
                transform=transform_test, 
                mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = usgs_dataset(
                dataset=self.dataset, 
                full_dataset=self.full_train_data,
                transform=transform_test, 
                mode='all')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)       
            return eval_dataset, eval_loader

    def check_noisy(self, pred):
        pred_clean_idx = pred.nonzero()[0]
        pred_noise_idx = (1-pred).nonzero()[0]
        self.noise_idx, self.clean_idx = np.array(self.noise_idx), np.array(self.clean_idx)

        correct_noisy = len(np.intersect1d(pred_noise_idx, self.noise_idx)) / len(self.noise_idx)
        correct_clean = len(np.intersect1d(pred_clean_idx, self.clean_idx)) / len(self.clean_idx)
        print("Percent noisy correct:", correct_noisy, "Percent clean correct:", correct_clean)

    def get_noisy_class_counts_train(self):
        return Counter(self.full_train_data.targets)

    def get_clean_class_counts_train(self):
        return Counter(self.orig_train_labels)
