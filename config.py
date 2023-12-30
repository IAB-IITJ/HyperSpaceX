import numpy as np
import torch
import random
import math

device = 'cuda:2'

epochs = 140 #180 #100 #cifar100 #15 #mnist
thr_epoch = 100 #120-use #130 #100#2 #-(none) #5#15 #12 #30 #20 ##95
thr_arc_epoch = 35 #20-use #70 #50 #30 #5 #5 #20 #2 ##45


lr = 0.001 #-sgd #0.001 #-adam #0.025 #cifar100 #0.01 #mnist
batch_size = 64 #32#64 #cifar100 #32 #mnist
weight_decay = 5e-4 
momentum = 0.9
step_size = 15

emb_size = 2 #512 #2
num_classes = 10 #200 #10575 #20
s_init = sorted(list(np.arange(5,101,5).astype('float')/10.0),reverse=False) #[5.0,50.0] 
# s1 = np.arange(5,(num_classes)*5+1,5).astype('float')/np.sqrt(2)
# random.shuffle(s1)     
# s2=-1*s1
# s=torch.stack((torch.tensor(s1),torch.tensor(s2))).to(device)
# s_init=s[0]*np.sqrt(2)

'''
s_init= np.arange(5,(num_classes)*5+1,5)
random.shuffle(s_init)
theta = np.arange(0,360,18)
theta=(math.pi*theta)/180
x=s_init*np.cos(theta)
y=s_init*np.sin(theta)
s=torch.stack((torch.tensor(x),torch.tensor(y))).to(device)
'''

num_workers = 4 #8
pin_memory = True
face = "clip"#"arcface_r18" #"arcface_ir50" #"EmbedderCifar10" #"arcface_ir50" #"arcface_ir100"  #"arcface_r18" 

if face == "EmbedderCifar10":
    resize = (32,32)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    fc_scale =  None 
elif face == "arcface_r18":
    resize = (128,128)
    mean = [0.5]
    std = [0.5] 
    fc_scale = 8 * 8
elif face == "arcface_ir50" or face == "arcface_ir100" or face == "arcface_ir18": 
    resize = (64,64)#(112,112) 
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    fc_scale =  7 * 7 
