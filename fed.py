from copy import deepcopy
import cvxpy as cp
import os
import random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset,DataLoader
import torch.utils.data

import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter, defaultdict
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
# tensorboard --logdir=logs/
class CustomSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[idx] for idx in indices]
    
    
def avg_model(model_dicts,weight):
    
    weight = np.array(weight)
    weight = weight/sum(weight)
    model_dict = deepcopy(model_dicts[0])
    for k in model_dict.keys():
        model_dict[k] = 0
        for i, model in enumerate(model_dicts):
            model_dict[k] += model[k]*weight[i]
    return model_dict

def dirichlet(
    targets, client_num: int, alpha: float,label_num:int
    ) -> Tuple[List[List[int]], Dict]:
    
    stats = {}
    targets_numpy = np.array(targets, dtype=np.int32)
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]



    data_indices = [[] for _ in range(client_num)]
    for k in range(label_num):
        np.random.shuffle(data_idx_for_each_label[k])
        distrib = np.random.dirichlet(np.repeat(alpha, client_num))
        distrib = distrib / distrib.sum()
        distrib = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(
            int
        )[:-1]
        data_indices = [
            np.concatenate((idx_j, idx.tolist())).astype(np.int64)
            for idx_j, idx in zip(
                data_indices, np.split(data_idx_for_each_label[k], distrib)
            )
        ]

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    data_indices

    return data_indices, stats

########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f'[SEED] ...seed is set: {seed}!')
    
MEAN = {
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "emnist": [0.1736],
    "fmnist": [0.286],
    "femnist": [0.9637],
    "medmnist": [124.9587],
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    "covid19": [125.0866, 125.1043, 125.1088],
    "celeba": [128.7247, 108.0617, 97.2517],
    "synthetic": [0.0],
    "svhn": [0.4377, 0.4438, 0.4728],
    "tiny_imagenet": [122.5119, 114.2915, 101.388],
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    "domain": [0.485, 0.456, 0.406],
}

STD = {
    "mnist": [0.3015],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "emnist": [0.3248],
    "fmnist": [0.3205],
    "femnist": [0.155],
    "medmnist": [57.5856],
    "medmnistA": [62.3489],
    "medmnistC": [58.8092],
    "covid19": [56.6888, 56.6933, 56.6979],
    "celeba": [67.6496, 62.2519, 61.163],
    "synthetic": [1.0],
    "svhn": [0.1201, 0.1231, 0.1052],
    "tiny_imagenet": [58.7048, 57.7551, 57.6717],
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    "domain": [0.229, 0.224, 0.225],
}

class MyOwnNet(nn.Module):
    def __init__(self,class_num):
        super(MyOwnNet, self).__init__()
        
        self.conv1=nn.Conv2d(3,32,5)
        self.conv1_bn=nn.BatchNorm2d(32)
        
        self.conv2=nn.Conv2d(32,32,5)
        self.conv2_bn=nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 5* 5, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        
        self.pool=nn.AvgPool2d(2,2)
        #self.dropout25 = nn.Dropout2d(p=0.25)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2_1 = nn.Linear(256, 256)

        # self.fc3 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(256, class_num)
        
    def forward(self,x):
        
        x=(self.pool(F.leaky_relu(self.conv1_bn(self.conv1(x)))))

        x=(self.pool(F.leaky_relu(self.conv2_bn(self.conv2(x)))))     

        x = x.reshape(-1, 32 * 5* 5)   
        
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)))
        x = F.leaky_relu(self.fc2_1(x))

        x = self.fc3(x)
        
        return x 
    
class SimpleCNN(nn.Module):
    
    def __init__(self,class_num):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 0.95 ^ 100 * 500 = 2 
def imbalance_dataset(dataset,imbalance):
    idx = defaultdict(list)
 
    for index,(x,y) in enumerate(dataset):
        idx[y].append(index)
    ret = []
    r = 1
    for i in range(len(idx.keys())):
        ret+= idx[i][0:(int)(len(idx[y])*r)]
        r *= imbalance
    logger.debug(f"len={len(ret)}")   
    print('划分结束')
    return CustomSubset(dataset,ret)
    
def train(model, trainloader, criterion, optimizer, device):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(trainloader), accuracy

def test(model, testloader, criterion, device):
    
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return test_loss / len(testloader), accuracy
def save_checkpoint(model,optimizer,epoch):
    checkpoint = {
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch
    }
    if not os.path.isdir("./models/checkpoint"):
        os.mkdir("./models/checkpoint")
    torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' %(str(epoch)))


def train_on_onedataset():

    batch_size = 24
    device = "cpu"

    transform_train = transforms.Compose([
 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
    writer=SummaryWriter("./logs")


    for imbalance in [1,0.98,0.96,0.94,0.92]:
        subset = imbalance_dataset(trainset,imbalance)
        model = MyOwnNet()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                            shuffle=True)
        for epoch in range(200):
            trainloss,trainacc = train(model, trainloader, criterion, optimizer, device)
            testloss,testacc = test(model,testloader,criterion,device)
            # writer.add_scalars(f"imbalance={imbalance}_loss",{"trainloss":trainloss,"testloss":testloss},epoch)
            writer.add_scalars(f"loss",{f"trainloss_{imbalance}":trainloss,f"testloss_{imbalance}":testloss},epoch)
            # writer.add_scalars(f"imbalance={imbalance}_acc",{"trainacc":trainacc,"testacc":testacc},epoch)
            writer.add_scalars(f"acc",{f"trainacc_{imbalance}":trainacc,f"testacc_{imbalance}":testacc},epoch)
            logger.info(f"epoch({epoch}):acc={trainacc}/{testacc},\t|loss={trainloss}/{testloss}")
    writer.close()
    
def sampling_by_distribution(clients,total_counter,args):
        logger.info("counter sampling....")
        counters = None
        for client in clients:
            if counters is None:
                counters = np.array(client.get_counter())
            else:
                counters = np.vstack([counters,np.array(client.get_counter())])
        
        # total_counter = np.sum(counters,axis=0)
        # # total_counter = np.ones(10)
        total_counter  = np.array(total_counter)
        total_counter = total_counter/sum(total_counter)
         
        
      
        matrix = None
        for i in range(counters.shape[0]):
            if matrix is None:
                matrix = counters[i,:]/sum(counters[i,:])
            else:
                matrix = np.vstack([matrix,counters[i,:]/sum(counters[i,:])])

        matrix = matrix.transpose()
  
        x = cp.Variable(matrix.shape[1])
        A = np.ones(len(clients))
        constrains = [0<=x,x<=1,A@x==1]
      
        objective = cp.Minimize(cp.sum_squares(matrix@x-total_counter))
    
        prob = cp.Problem(objective=objective,constraints=constrains)
        prob.solve()
    
        probablity =  x.value
        distance = prob.value
        for i,p in enumerate(probablity):
            if p <=0:
                probablity[i] = 1e-6
        probablity = probablity/sum(probablity)
        logger.info(f"distance={distance:<6.4f},prob = {probablity}")
        sampled_num = max(1,int(args['sampling_ratio']*len(clients)))
        sampled = np.random.choice(clients,sampled_num,replace=False,p=probablity)
        logger.info(f"sampled clients:{sampled}")
        return sampled

def saveargs(args):
    with open(args['tensorboard_logdir']+'/args.yaml', 'w') as f:
        yaml.dump(args, f)

if __name__ == '__main__':

    transform_train = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    data_indices, stats = dirichlet(trainset.targets,10,0.1,100)
    print(data_indices,stats)
