
import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#from resnet import resnet_face18
from iresnet import iresnet50
import os

import torch.nn.functional as F
import numpy as np
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from CLIP import clip
import math

# config
device = 'cuda:1'

emb_size = 2
loss_name='proposed'
model_name='clip'
dataset_name = 'mnist'
save = False
save_name = 'models/mnist/cr_fashion_2_r101_g5'
plots=True
plots_save_name ='plots/clip_mnist/dist_arc/' #'plots/clip_mnist/div_center/'plots/clip_mnist/dist_arc/'
layers_freeze=90
gap=10
num_classes = 10 
batch_size = 64 
epochs = 150
lr = 0.001#2.048e-3 #0.0001
weight_decay = 5e-4 
momentum = 0.9
resize   = (112,112)
mean = [0.5,0.5,0.5] 
std = [0.5,0.5,0.5] 
num_workers = 4
pin_memory = True

print("dataset:",dataset_name)
print('loss:',loss_name)
print("emb:",emb_size)
print("gap",gap)
print("model",model_name)

def log_softmax(x,y): 
    return x - y.exp().sum(-1).log().unsqueeze(-1)

def nll(input, target): 
    return (((-input[range(target.shape[0]), target]))).mean()

def Contrastive_Center(targ, x, centers):
    bs = x.shape[0]
    expanded_centers = centers.expand(bs, -1, -1)
    expanded_hidden = x.expand(num_classes, -1, -1).transpose(1, 0)
    distance_centers = (expanded_centers - expanded_hidden).pow(2).sum(dim=-1)
    return distance_centers
    
    distances_same = distance_centers.gather(1, targ.unsqueeze(1))
    intra_distances = distances_same.sum()
    print(intra_distances.shape)
    epsilon = 1e-6
    inter_distances = distance_centers.sum().sub(intra_distances) + epsilon
    print(inter_distances.shape)
    #loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / (inter_distances + epsilon) / 0.1
    return intra_distances, inter_distances
if gap==10:
    s1 = torch.tensor(np.arange(5,101,10).astype('float')).to(device)
elif gap==5:
    s1 = torch.tensor(np.arange(5,51,5).astype('float')).to(device)

mse = nn.MSELoss()
class ArcFaceClassifier(nn.Module):
    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x, targ):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)

        news=s1-9
        W1  = news.reshape(-1,1)*F.normalize(self.W.T, dim=1)
        R = -W1[targ] + x
        W1_norm = -F.normalize(W1.T, dim=0)
        R_norm = F.normalize(R)

        W1  = (s1+5).reshape(-1,1)*F.normalize(self.W.T, dim=1)
        
        dist = Contrastive_Center(targ, x, W1)
        dis_norm = (torch.norm(W1[targ],dim=1)-torch.norm(x,dim=1))**2
        # dis_ = (dist,dis_norm)


        # dis1=(s1[targ]-torch.norm(x,dim=1))**2
        # dis1 = (dis1 - torch.min(dis1)) / (torch.max(dis1) - torch.min(dis1))
        dis1=(dist,dis_norm)

        return x_norm @ W_norm, R_norm @ W1_norm, dis1, x, self.W  
    
cross_entropy = nn.CrossEntropyLoss()
def arcface_loss(cosine, resultant_cosine,dist,x,W,targ,m=0.5,epoch=0):
    # return cross_entropy(cosine, targ)
    
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    arcosine = cosine.arccos()
    arcosine += F.one_hot(targ, num_classes = num_classes) * m
    cosine2 = arcosine.cos()
    cosine2 = cosine2*1.0
    if loss_name=='arcface':
        return F.cross_entropy(cosine2, targ)

    resultant_cosine = resultant_cosine.clip(-1+1e-7, 1-1e-7) 
    # arcosine = resultant_cosine.arccos(
    # arcosine+=math.pi/2
    # resultant_cosine=arcosine.cos()
    resultant_cosine *= F.one_hot(targ, num_classes = num_classes)

    dist1 = dist[0]
    dis_norm = dist[1]

    cosine_lower = 2.0*cosine2-0.005*(dist1-(F.one_hot(targ, num_classes = num_classes) * dist1))#-resultant_cosine#+dist.reshape(-1,1)*F.one_hot(targ, num_classes = num_classes)
    cosine_upper = 2.0*cosine2-0.005*dist1#.reshape(-1,1)*F.one_hot(targ, num_classes = num_classes)
    #cosine_upper = 2.0*cosine2-resultant_cosine#-dist1.reshape(-1,1)*F.one_hot(targ, num_classes = num_classes)
    #cosine_upper = cosine2-resultant_cosine-dist#0.001*(dist)#/((dist-F.one_hot(targ, num_classes = num_classes) * dist).sum(dim=1,keepdim=True))) #(dis1.reshape(-1,1)*F.one_hot(targ, num_classes = num_classes))
    #dis_norm = 0.001*dist[1].mean()
    # if epoch>=20:
    #     cosine_lower = 5.0*cosine2-0.005*(dist1-(F.one_hot(targ, num_classes = num_classes) * dist1))
    #     cosine_upper = 5.0*cosine2-resultant_cosine-0.005*dist1 #dist.reshape(-1,1)*F.one_hot(targ, num_classes = num_classes)
        #dis_norm = 0.001*dis_norm.mean()
    #print(cos)
    loss = log_softmax(cosine_upper,cosine_lower)
    
    
    loss = nll(loss,targ)#+ dis_norm #0.01*mse(x,(s1.reshape(-1,1)*F.normalize(W.T, dim=1))[targ].float())
    return loss

class ArcfaceClipFeatures(nn.Module):
    def __init__(self,clip_m):
        super().__init__()
        self.clip_model_image_features = clip_m
        clip.model.convert_weights(self.clip_model_image_features)
        self.linear1 = nn.Linear(512,emb_size,bias=False)
        if loss_name=='cross':
            self.linear2 = nn.Linear(emb_size,num_classes)

    def forward(self, x):
        x = self.clip_model_image_features(x)
        x = x.float()
        #print(x.shape)
        x = self.linear1(x)

        if loss_name=='cross':
            x = self.linear2(x)
        return x

class ResnetModelClassifier(nn.Module):
    def __init__(self,model_resnet):
        super().__init__()
        self.model_features = model_resnet.to(device)
        
        self.linear1 = nn.Linear(512,emb_size,bias=False).to(device)
        if loss_name=='cross':
            self.linear2 = nn.Linear(emb_size,num_classes).to(device)

    def forward(self, x):
        x=self.model_features(x)
        
        x = self.linear1(x)
        if loss_name=='cross':
            x = self.linear2(x)
        return x

def load_model(model):
    checkpoint = torch.load('iresnet50_imagenet.pth',map_location=device) 
    mystatedict={}
    for key in checkpoint.keys():
        if "fc" in key:
            continue
        else:
            m=key.replace("module.","")
            mystatedict[m] = checkpoint[key]

    model.load_state_dict(mystatedict,strict=False)
    return model


clip_m, preprocess = clip.load('ViT-B/32', device)
if model_name=='iresnet50':
    #model = ResnetModelClassifier(iresnet50(pretrained=False)).to(device)
    model = ResnetModelClassifier(load_model(iresnet50(pretrained=False)).to(device))
elif model_name=='clip':
    model = ArcfaceClipFeatures(clip_m).to(device)
print(model)
#
classifier = ArcFaceClassifier(emb_size, num_classes).to(device)

if loss_name=='cross':
    criterion=nn.CrossEntropyLoss()
else:
    criterion = arcface_loss


def convert_models_to_fp32(mod): 
    for p in mod.parameters(): 
        p.data = p.data.float() 
        if p.requires_grad == True:
            p.grad.data = p.grad.data.float() 

c=0
for name,param in model.named_parameters():
    c+=1
    print(c,name)
    if c<=layers_freeze: #72
        
        param.requires_grad=False

optimizer_model = optim.SGD(model.parameters(), lr=0.001,weight_decay = weight_decay,momentum=momentum)
optimizer_classifier = optim.SGD(classifier.parameters(), lr=0.001,weight_decay = weight_decay,momentum=momentum)


# eval-loop
test_loss = []
best_accuracy1 = 0.0
best_accuracy2 = 0.0
best_epoch1 = 0
best_epoch2 = 0

def train_test(dloader,epoch,phase='train'):
    global best_accuracy1
    global best_accuracy2
    global best_epoch1
    global best_epoch2
    if phase=='test':
        model.eval()
        classifier.eval()
    elif phase=='train':
        model.train()
        classifier.train()
    teloss = 0.0
    correct1 = 0
    correct2 = 0

    for (data,label) in tqdm(dloader,ncols=45):
        data = data.float().to(device)
        label = label.long().to(device)
        embedding = model(data)
        if loss_name=='cross':
            los=criterion(embedding,label)
            pred1=torch.argmax(embedding,1)
        elif loss_name=='arcface':
            angles, resultant_cosine, resultant_vector,x,W = classifier(embedding,label)
            pred1=torch.argmax(angles,1)
            los = criterion(angles,resultant_cosine,resultant_vector,x,W,label,epoch)
        elif loss_name=='proposed':
            angles, resultant_cosine, resultant_vector,x,W = classifier(embedding,label)
            s_new = torch.ones((angles.shape[0],1)).to(device)*s1.reshape(1,-1)
            emb_norm = torch.norm(embedding,dim=1)
            distances = (s_new-emb_norm.reshape(-1,1)*torch.ones((1,num_classes)).to(device))**2
            distances = F.softmax(distances,dim=1)
            pred=angles-distances
            pred2 = torch.argmin(resultant_vector[0],1)
            correct2 += (pred2==label).sum().item()
            los = criterion(angles,resultant_cosine,resultant_vector,x,W,label,epoch)
            pred1 = torch.argmax(pred,1)
        correct1 += (pred1==label).sum().item()

        teloss += los.item()
        if phase=='train':
            optimizer_classifier.zero_grad()
            optimizer_model.zero_grad()
            los.backward()
            optimizer_classifier.step()
            
            optimizer_model.step()
            if model_name=='clip':
                convert_models_to_fp32(model.clip_model_image_features)
                clip.model.convert_weights(model.clip_model_image_features)

    test_loss.append(teloss/len(dloader))
    accuracy1 = 100 * correct1/(batch_size*len(dloader))
    accuracy2 = 100 * correct2/(batch_size*len(dloader))
    if loss_name=='proposed':
        print("epoch=",epoch,'Train loss = ',round((teloss/len(dloader)),2),'Accuracy1 = ',round(accuracy1,2),'Accuracy2 = ',round(accuracy2,2))
    else:
        print("epoch=",epoch,'Train loss = ',round((teloss/len(dloader)),2),'Accuracy = ',round(accuracy1,2))
    if phase=='test':
        if accuracy1 >= best_accuracy1:
            best_accuracy1 = accuracy1
            if save==True:
                torch.save(model.state_dict(),save_name+'_model.pth')#'models/cifar10/pr_512_clip_g5.pth')
                torch.save(classifier.state_dict(),save_name+'_cls.pth')#'models/cifar10/pr_512_clip_g5.pth')
            best_epoch1 = epoch
        if accuracy2 >= best_accuracy2:
            best_accuracy2 = accuracy2
            if save==True:
                torch.save(model.state_dict(),save_name+'_model.pth')#'models/cifar10/pr_512_clip_g5.pth')
                torch.save(classifier.state_dict(),save_name+'_cls.pth')#'models/cifar10/pr_512_clip_g5.pth')
            best_epoch2 = epoch
        print("best_accuracy1:",round(best_accuracy1,2),"best_epoch2:",best_epoch1)
        if loss_name=='proposed':
            print("best_accuracy2:",round(best_accuracy2,2),"best_epoch2:",best_epoch2)



def get_embs(model,dl):
    model.eval()
    embs = []
    ys = []
    for bx,by in tqdm(dl,ncols=45):
        with torch.no_grad():
            emb = model(bx.float().to(device))
            x = emb
            embs.append(x.detach().cpu().numpy())
            ys.append(by.detach().cpu().numpy())

    embs = np.concatenate(embs)
    ys = np.concatenate(ys)

    return embs,ys

def plot_embs(embs, ys, w_norm, ax):
    color_list = ['black','dimgrey','brown','red','orangered','gold','yellow','lawngreen','green','aquamarine','cyan','blue','darkviolet','magenta','orange','olive','palevioletred','midnightblue','coral','teal']

    for k in range(num_classes):
        w_e = s1[k].detach().cpu().numpy()*w_norm[k]
        e = embs[ys==k]

        ax.scatter(e[:,0], e[:,1], s=2, alpha=.8,color=color_list[k])
        #ax.plot((0,np.mean(e,0)[0]),(0,np.mean(e,0)[1]),color=color_list[k],linestyle='--',linewidth=2) 
        
        if loss_name=='proposed':
            ax.plot((0,w_e[0]),(0,w_e[1]),color=color_list[k],linestyle='--',linewidth=2) 
            dist2d = np.linalg.norm(np.array([0,0])-np.array([w_e[0],w_e[1]]))
            ci = plt.Circle((0,0),dist2d,fill = False,color=color_list[k],linestyle='--')
            ax.add_artist(ci)
            ax.set_xlim([-100,100])
            ax.set_ylim([-100,100])

def main(): 
    root = os.path.expanduser("~/.cache")
    if dataset_name=='cifar10':
        train_dataset = datasets.CIFAR10(root=root, train=True, transform=preprocess, download=True)
        val_dataset = datasets.CIFAR10(root=root, train=False, transform=preprocess, download=True)
    elif dataset_name=='mnist':
        train_dataset = datasets.MNIST(root=root, train=True, transform=preprocess, download=True)
        val_dataset = datasets.MNIST(root=root, train=False, transform=preprocess, download=True)
    elif dataset_name=='fashion':
        train_dataset = datasets.FashionMNIST(root=root, train=True, transform=preprocess, download=True)
        val_dataset = datasets.FashionMNIST(root=root, train=False, transform=preprocess, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size,shuffle=False)

    for epoch in range(0,epochs):
        train_test(train_dataloader,epoch,'train')
        if plots==True:     
            classifier.eval()   
            # W = torch.cat([p for n,p in classifier.named_parameters() if 'linear2' in n]) 
            # W1_norm = W.detach().cpu().numpy()

            W = torch.cat([p for n,p in classifier.named_parameters() if 'W' in n]).T
            W1_norm = (F.normalize(W.T,dim=0).T).detach().cpu().numpy()
            embs, ys = get_embs(model,train_dataloader)
            _,(ax1)=plt.subplots(1,1, figsize=(10,10))
            plot_embs(embs, ys, W1_norm, ax1)
            plt.savefig(plots_save_name+'train/epoch'+str(epoch)+'.jpg') 
            plt.close()
            
        train_test(val_dataloader,epoch,'test')
        if plots==True:   
            classifier.eval()    
            # W = torch.cat([p for n,p in classifier.named_parameters() if 'linear2' in n]) 
            # W1_norm = W.detach().cpu().numpy()

            W = torch.cat([p for n,p in classifier.named_parameters() if 'W' in n]).T
            W1_norm = (F.normalize(W.T,dim=0).T).detach().cpu().numpy()
            
            embs, ys = get_embs(model,val_dataloader)
            _,(ax1)=plt.subplots(1,1, figsize=(10,10))
            plot_embs(embs, ys, W1_norm, ax1)
            plt.savefig(plots_save_name+'test/epoch'+str(epoch)+'.jpg')  #clip_mnist/div_center-normal #fashion_new(run1) - resultant + (||x|| - ||sW||) with bias *10.0 #weight_arc_dist - resultant + (||x|| - ||sW||) #cifar10_2_gray(run2) -resultant + (||x|| - ||sW||) with bias variable weightage # mnist all ir50 #dist_arc - +resultant and all ir50 mnist
main()
