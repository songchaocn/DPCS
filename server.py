
from copy import deepcopy
import random
from typing import List
from loguru import logger
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from client import FedAvgClient
from torch.utils.tensorboard import SummaryWriter

from fed import MyOwnNet, avg_model, dirichlet, imbalance_dataset, sampling_by_distribution, saveargs, set_seed, test

def client_data_inc(clients:FedAvgClient):
    for cli in clients:
        cli.inc_by_ratio(cli.args['inc_ratio'])
        
def compute_distance(cnts:List[List],targert)->float:
            cnt = len(cnts[0])*[0]
            for li in cnts:
                for index,i in enumerate(li):
                    cnt[index]+=i
            x1 =np.array(cnt)
            x1 = x1/sum(x1)
            x2 = np.array(targert)
            x2 = x2/sum(x2)
            return np.linalg.norm(x1-x2,2)  
         
def fed_training(args):

    writer=SummaryWriter(args['tensorboard_logdir'])
    saveargs(args)
   
    transform_train = transforms.Compose([
       
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    imbalance_trainset = imbalance_dataset(trainset,args['imbalance'])

    client_num = args['client_num']
    alpha=args['alpha']
    class_num=args['class_num']
    data_indices, stats = dirichlet(imbalance_trainset.targets,client_num,alpha,class_num)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['train_batch_size'],
                                         shuffle=False)
    client_datasets=[torch.utils.data.Subset(imbalance_trainset,data_indices[i]) for i in range(client_num)]
    clients = []
    for idx,dataset in enumerate(client_datasets):
        clients.append(FedAvgClient(dataset,testset,MyOwnNet(args['class_num']),args,idx))
    rounds = args['rounds']
    global_model = MyOwnNet(args['class_num'])
    total_counter=[0]*args['class_num']
    for x,y in testset:
        total_counter[y]+=1
        
    for r in range(rounds):
        total_loss=0
        total_sample=0
        
        sampled_num = max(1,int(args['sampling_ratio']*len(clients)))
        sampled_clients = np.random.choice(clients,sampled_num,replace=False)
        # sampled_clients = sampling_by_distribution(clients,total_counter,args)
        logger.info(f"sampled:{[cli.id for cli in sampled_clients]}")
      
        client_data_inc(clients)
        cnts=[]
        for cli in sampled_clients:
            cli.model.load_state_dict(global_model.state_dict())
            tloss,tsample = cli.train()
         
            total_loss += tloss
            total_sample += tsample
        model_dicts= [cli.model.state_dict() for cli in sampled_clients]
        model_dict= avg_model(model_dicts,[1 for _ in range(len(model_dicts))])
        global_model.load_state_dict(model_dict)
        if (r+1)%5==0:
            cnts=[]
            for cli in sampled_clients:
                cnts.append(cli.get_counter())
            
            distance = compute_distance(cnts,[1 for _ in range(args['class_num'])])
            im = args['imbalance']
            writer.add_scalars(f"dictance",{f'distance_{im}':distance},r)
            logger.debug(f"distance:{distance}")
            
            criterion = torch.nn.CrossEntropyLoss()
            testloss,testacc = test(global_model,testloader,criterion,args['device'])
         
            
            writer.add_scalars(f"acc",{f'test_acc_{im}':testacc},r)
            writer.add_scalars(f"loss",{f'train_loss_{im}':total_loss/total_sample},r)
            logger.info(f"round:{r},testacc={testacc}")
    writer.close()
    
    
if __name__ == '__main__':
    args=dict()
    

    device = "cuda:1"
    args['tensorboard_logdir']='./logs20241202lab2'
    
    args['class_num'] = 100 
    args['dataset'] ='cifar100'
    
    args['seed'] = 40
    args['device']=device
    
    args['client_num'] = 20
    args['alpha'] =10 
    args['local_lr']=0.001
    args['momentum'] = 0.9
    args['train_batch_size']=128
    args['local_epoch']=5
    
    args['rounds'] = 1000
    args['sampling_ratio'] = 0.1
    args['inc_ratio'] = 0.001
    args['init_ratio'] = 0.3
    
    args['imbalance'] =0.99 
    fed_training(args)