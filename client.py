from typing import Dict, List
from loguru import logger
import torch
import torch.utils
from torch.utils.data import DataLoader
import os
import sys

import torch.utils.data
from fed import test,train
from probables import BloomFilter, CountMinSketch

class CMSketch:
    def __init__(self,args:Dict):
        self.sketch = CountMinSketch(20,4)
        self.args = args
        
    def add_element(self,element):
        self.sketch.add(str(element),1)
    
    def get_counter(self)->List[int]:
        class_num = self.args['class_num']
        distri = [0]*class_num
        for i in range(class_num):
            cnt = self.sketch.check(str(i))
            distri[i] += cnt
        return distri
    
class Bfilter:
    def __init__(self,args:Dict):
        self.blm = BloomFilter(est_elements=args['class_num'],false_positive_rate=0.05)
        self.args = args
    def add_element(self, element):
        self.blm.add(str(element))
        
    def get_counter(self):
        class_num = self.args['class_num']

        counter = [0]*class_num
        for i in range(class_num):
            if self.blm.check(str(i)):
                counter[i] = 1
        return counter
    
class Counter:
    def __init__(self,args:Dict):
        self.args=args
        self.counter = [0]*args['class_num']
    
    def add_element(self,element:int):
        self.counter[element] += 1
        
    def get_counter(self)->List[int]:
        return self.counter
        
        
class FedAvgClient:
    def __init__(self,train_dataset=None,test_dataset=None,model=None,args=None,id=None) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.args  = args
        self.id=id
        
        # self.counter = BloomFilter(self.args['class_num'])
        self.counter = Bfilter(args)
        # self.counter = Counter(args)
        # self.counter = CMSketch(args)
        # self.counter = Counter(args)
        
        
        self.cur_data=[]
        self.total_data =[i for i in range(len(train_dataset))]
        self.cur_pos= 0
        
        self.inc_by_ratio(inc_ratio=args['init_ratio'])

    
    def inc_by_ratio(self,inc_ratio:float):
        inccnt = int(inc_ratio*len(self.total_data))
        inccnt = max(inccnt,1)
        self.inc_by_cnt(inccnt)
        
    
    def get_counter(self):
        return self.counter.get_counter()
    
    def get_train_sample_num(self):
        return len(self.train_dataset)

    def inc_by_cnt(self,cnt):
        nextp = cnt+self.cur_pos
        nextp = min(nextp,len(self.total_data))
        for i in range(self.cur_pos,nextp):
            self.cur_data.append(i)
            self.counter.add_element(self.train_dataset[i][1])
        self.cur_pos =nextp

    def get_model_dict(self):
        return self.model.state_dict()

    def train(self):
        # logger.info(f"{self}:traing...train_set[{len(self.train_dataset)}]")

        lr = self.args['local_lr']
        train_batch_size = self.args['train_batch_size']
        momentum = self.args['momentum']
        local_epoch = self.args['local_epoch']
        trainset = torch.utils.data.Subset(self.train_dataset,self.cur_data)
        dataloader = DataLoader(trainset,train_batch_size,True,drop_last=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)
        model = self.model
        model.to(self.args['device'])
        model.train()
        
        cnt = 0
        total_loss = 0
        total_sample = 0
        total_correct = 0
        logger.debug(f"{self}:traing...train_set[{len(trainset)}]")
        iterator = iter(dataloader)
        
        for epoch in range(local_epoch):
            try:
                x,y = next(iterator)
            except:
                iterator = iter(dataloader)
                x,y = next(iterator)
         
            x = x.to(self.args['device'])
            y = y.to(self.args['device'])
            model.zero_grad()
            output = model(x)
            loss = criterion(output,y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sample += len(y)
               
        return total_loss,total_sample


    
    def __repr__(self) -> str:
        return f"cli-{self.id}[{len(self.train_dataset)}-{len(self.test_dataset)}]"
