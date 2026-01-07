import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.io as io
from PIL import Image
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset, random_split
from simplecv.data import preprocess
import cv2
from tqdm import tqdm
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

os.chdir('D:/SMViT')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mask_dirs='data/land_mask.mat'
mask_data = io.loadmat(mask_dirs)
land_mask = mask_data['land_mask'][21:,60:956]
land_mask=land_mask>0

datas = []
image_base = 'data/train/input/'   
annos_base = 'data/train/label/'  

ids_ = [v.split('.')[0] for v in os.listdir(image_base)]
ids_ = sorted([int(i) for i in ids_])
ids_ = [str(i) for i in ids_]

for id_ in ids_:
    img_pt0 = os.path.join(image_base, '{}.npy'.format(id_))
    img_pt1 = os.path.join(annos_base, '{}.npy'.format(id_))
    if os.path.exists(img_pt0) and os.path.exists(img_pt1):
        datas.append((img_pt0, img_pt1))
    else:
        print(f"Warning: Missing file for id {id_}")
        continue

print(f"Total samples: {len(datas)}")

class MyDataset(Dataset): 
    def __init__(self, transform=None, data=None):
        super(MyDataset, self).__init__()
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, mask_path = self.data[index]
        
        img = np.load(img_path)[:, :, :]
        img = torch.from_numpy(img).type(torch.FloatTensor)
        
        mask = np.load(mask_path)[:, :]
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        
        return img, mask
    
    def __len__(self):
        return len(self.data)

full_dataset = MyDataset(data=datas)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

from src.model.SMViT import SMViT
net = SMViT(
    encoder_name="resnet34",
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
) 
net.to(device)

criterion = nn.HuberLoss()
maecriterion = nn.L1Loss()
msecriterion = nn.MSELoss()

optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

mask = torch.from_numpy(land_mask).unsqueeze(0).unsqueeze(1).to(device)

best_val_loss = float('inf')
patience_counter = 0
patience = 10
best_model_path = 'weight/best_model.pth'

# 创建保存目录
os.makedirs('weight', exist_ok=True)

def validate(model, val_loader, device):
    """在验证集上评估模型"""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            y_pred = model(inputs)
            mask_ = mask.repeat(labels.shape[0], 1, 1, 1)
            label_mask = mask_.squeeze(1)
            masked_pred = y_pred[mask_]
            masked_labels = labels[label_mask]
            
            loss = msecriterion(masked_pred.squeeze(), masked_labels.squeeze()) + \
                   msecriterion(y_pred.squeeze(), labels.squeeze())
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

train_losses = []
val_losses = []
net.train()
for epoch in range(100):
    epoch_loss = 0.0
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/200 [Train]', leave=False)
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        y_pred = net(inputs)
        mask_ = mask.repeat(labels.shape[0], 1, 1, 1)
        label_mask = mask_.squeeze(1)
        masked_pred = y_pred[mask_]
        masked_labels = labels[label_mask]
        
        loss = msecriterion(masked_pred.squeeze(), masked_labels.squeeze()) + \
               msecriterion(y_pred.squeeze(), labels.squeeze()) + 0.1 * maecriterion(y_pred.squeeze(), labels.squeeze())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        train_pbar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    avg_val_loss = validate(net, val_dataloader, device)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)
  
    print(f'Epoch {epoch+1}: '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        }, best_model_path)
        print(f'Model improved. Saved best model to {best_model_path}')
    else:
        patience_counter += 1
        print(f'No improvement for {patience_counter} epoch(s)')
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

print('Finished Training')

checkpoint = torch.load(best_model_path)
net.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded best model from epoch {checkpoint["epoch"]} with val loss {checkpoint["best_val_loss"]:.4f}')


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

class testDataset(Dataset): 
    def __init__(self,transform=None,data=None):
                            
        super(testDataset,self).__init__()
        self.imgs = []
        self.transform = transform
        self.data=data   
    def __getitem__(self, index):
        
        fn = self.data[index][0];label= self.data[index][1]
        img = np.load(fn)[:,:,:]
        img=torch.from_numpy(img).type(torch.FloatTensor)
        mask = np.load(label)[:,:]
        mask=torch.from_numpy(mask).type(torch.FloatTensor)
        return img,mask
    def __len__(self):
        return len(self.data)


test_datasets=testDataset(data=datas)
test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)

def reg_metric(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return (mae,mse,rmse,r2)

net.eval()

reslist=[]
for i,batch in enumerate(test_dataloader):
    inputs1 = batch[0].to(device)
    labels1 = batch[1].to(device)
    y_pred1 = net(inputs1).squeeze()
    tmp1=y_pred1.detach().cpu().numpy()
    tmp1[~land_mask]=0
    
    mask1=labels1.detach().cpu().numpy()
    pridiction = (tmp1*255).astype(np.uint8)
    target = (mask1[0]*255).astype(np.uint8)

    res1=reg_metric(target,pridiction)

    reslist.append(res1)
    imgpath='data/result/'
    imgres=Image.fromarray((tmp1*255).astype(np.uint8))
    imgres.save(imgpath+str(i)+'.png')
    imgtrue=Image.fromarray((mask1[0]*255).astype(np.uint8))
    imgtrue.save(imgpath+str(i)+'_true.png')
print(reslist)

dirs='data/result/'
dirlist=os.listdir(dirs)


all_pridiction=np.array([])
all_true=np.array([])

def calc_mae(i,dirs):
    pridiction=Image.open(dirs+str(i)+'.png')
    pridiction=np.array(pridiction)
    
    pridiction=pridiction.reshape(-1,1)
    truelabel=Image.open(dirs+str(i)+'_true.png')
    truelabel=np.array(truelabel)
    truelabel=truelabel.reshape(-1,1)

    ignored_mask = np.zeros(truelabel.shape[:2], dtype=bool)
    ignored_mask[truelabel == 0] = True
    truelabel = truelabel[~ignored_mask]
    pridiction = pridiction[~ignored_mask]
    return pridiction,truelabel

for i in range(466):
    pridiction,truelabel=calc_mae(i,dirs)
    all_pridiction=np.append(all_pridiction,pridiction)
    all_true=np.append(all_true,truelabel)
print('over')
ans=reg_metric(all_true/10.625,all_pridiction/10.625)
txtdirs=dirs+'result.txt'
with open(txtdirs, 'w') as f: 
    print('MAE:',ans[0],file=f)
    print('MSE:',ans[1],file=f)
    print('rmse:',ans[2],file=f)
    print('r2:',ans[3],file=f)

print('MAE:',ans[0])
print('MSE:',ans[1])
print('rmse:',ans[2])
print('r2:',ans[3])