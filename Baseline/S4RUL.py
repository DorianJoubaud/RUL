import torch
from torch import nn, Tensor
import math
import torch.optim as optim

from s4 import S4Block as S4 
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import random
import torch.nn.functional as F
import os
import wandb
import argparse


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_num]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)
    


class S4ModelForRUL(nn.Module):
    def __init__(self, d_input, d_model=512, n_layers=4, dropout=0.1, max_len=500):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.encoder = nn.Linear(d_input, d_model)
        self.bn_encoder = nn.BatchNorm1d(max_len)
        self.s4_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4(d_model, dropout=dropout, transposed=True))
        self.attention = ScaledDotProductAttention(d_model)
        self.decoder = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.encoder(src)
        src = self.bn_encoder(src)
        src = self.pos_encoder(src)
        src = src.transpose(1, 2)  # S4 expects [batch_size, d_model, seq_len]

        for layer in self.s4_layers:
            src, _ = layer(src)

        src, attention_weights = self.attention(src, src, src)  # Apply self-attention
        src = src.transpose(1, 2)  # Back to [batch_size, seq_len, d_model]
        src = self.dropout(src)
        output = self.decoder(src)
        return output

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


class load_data(Dataset):
    """
    root = new | old
    """
    def __init__(self, name, seq_len, root='new') -> None:
        super().__init__()
        data_root = "data/units/"
        if root == 'old':
            label_root = "data/labels/"
        elif root == 'new':
            label_root = "data/new_labels/"
        else:
            raise RuntimeError("got invalid parameter root='{}'".format(root))
        raw = np.loadtxt(data_root+name)[:,2:]
        lbl = np.loadtxt(label_root+name)/Rc
        l = len(lbl)
        if l<seq_len:
            raise RuntimeError("seq_len {} is too big for file '{}' with length {}".format(seq_len, name, l))
        raw, lbl = torch.tensor(raw, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)
        lbl_pad_0 = [torch.ones([seq_len-i-1]) for i in range(seq_len-1)] 
        data_pad_0 = [torch.zeros([seq_len-i-1,24]) for i in range(seq_len-1)]
        lbl_pad_1 = [torch.zeros([i+1]) for i in range(seq_len-1)] 
        data_pad_1 = [torch.zeros([i+1,24]) for i in range(seq_len-1)]
        self.data = [torch.cat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
        self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.data += [torch.cat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
        self.label = [torch.cat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
        self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.label += [torch.cat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
        self.padding = [torch.cat([torch.ones(seq_len-i-1), torch.zeros(i+1)],0) for i in range(seq_len-1)]   # 1 for ingore
        self.padding += [torch.zeros(seq_len) for i in range(seq_len-1, l)]
        self.padding += [torch.cat([torch.zeros(seq_len-i-1), torch.ones(i+1)],0) for i in range(seq_len-1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]


class load_all_data(Dataset):
    """
    root: new | old
    name: LIST of txt files to collect 
    """
    def __init__(self, name, seq_len) -> None:
        super().__init__()
        data_root = "data/units/"
        label_root = "data/new_labels/"
        lis = os.listdir(data_root)
        data_list = [i for i in lis if i in name]
        self.data, self.label, self.padding = [], [], []
        for n in data_list:
            raw = np.loadtxt(data_root+n)[:,2:]
            lbl = np.loadtxt(label_root+n)/Rc
            l = len(lbl)
            if l<seq_len:
                raise RuntimeError("seq_len {} is too big for file '{}' with length {}".format(seq_len, n, l))
            raw, lbl = torch.tensor(raw, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)
            lbl_pad_0 = [torch.ones([seq_len-i-1]) for i in range(seq_len-1)] 
            data_pad_0 = [torch.zeros([seq_len-i-1,24]) for i in range(seq_len-1)]
            lbl_pad_1 = [torch.zeros([i+1]) for i in range(seq_len-1)] 
            data_pad_1 = [torch.zeros([i+1,24]) for i in range(seq_len-1)]
            self.data += [torch.cat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
            self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.data += [torch.cat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
            self.label += [torch.cat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
            self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.label += [torch.cat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
            self.padding += [torch.cat([torch.ones(seq_len-i-1), torch.zeros(i+1)],0) for i in range(seq_len-1)]   # 1 for ingore
            self.padding += [torch.zeros(seq_len) for i in range(seq_len-1, l)]
            self.padding += [torch.cat([torch.zeros(seq_len-i-1), torch.ones(i+1)],0) for i in range(seq_len-1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]
    
    

def get_score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))

def get_pred_result(data_len, out, lb):
    pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
    for j in range(data_len):
        if j < seq_len-1:
            pred_sum[:j+1] += out[j, -(j+1):]
            pred_cnt[:j+1] += 1
        elif j <= data_len-seq_len:
            pred_sum[j-seq_len+1:j+1] += out[j]
            pred_cnt[j-seq_len+1:j+1] += 1
        else:
            pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += out[j, :(data_len-j)]
            pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += 1
    truth = torch.tensor([lb[j,-1] for j in range(len(lb[0])-seq_len+1)], dtype=torch.float)
    
    pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
    pred2 = pred_sum/pred_cnt
    pred2 *= Rc
    truth *= Rc
    return truth, pred2 

def train(data, model, loss_function, optimizer, seq_len, epochs, device, name, target):
    min_rmse = float('inf')
    for e in range(epochs):
        model.train()
        random.shuffle(data)
        train_data = load_all_data(data, seq_len=seq_len)  # Ensure this returns a dataset compatible with DataLoader
        total_loss = 0.0
        
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
       
        for train_data, train_label, train_padding in train_loader:
            
            train_data, train_label = train_data.to(device), train_label.to(device)
            optimizer.zero_grad()
            output = model(train_data).squeeze()  # Adjusted to pass only train_data
            
            
            loss = loss_function(output, train_label)
            
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            
            

        
       
           
        rmse = validate(model, seq_len, device, target, score = False) 
        wandb.log({"Epoch": e, "Loss": total_loss / len(train_loader), "ValLoss": rmse, 'optimizer': optimizer.param_groups[0]['lr'], 'scheduler': sch.get_last_lr()[0]})
        print(f"Epoch: {e}, Loss: {total_loss / len(train_loader)}, Val loss: {rmse}")
            
        
        if rmse < min_rmse:
            min_rmse = rmse
            torch.save(model.state_dict(), f'save/s4_{name}.pth')
        
        sch.step()
    
    return min_rmse

            
            
            
        
def validate(model, seq_len, device, val_data, testing = False, score = True):
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    with torch.no_grad():
        for i in val_data:
            pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
            valid_data = load_data(i, seq_len)  # Ensure this returns a dataset compatible with DataLoader
            valid_loader = DataLoader(valid_data, batch_size=1000, shuffle=False)
            
           
            for valid_data, valid_label, valid_padding in valid_loader:
               
                valid_data = valid_data.to(device)
                data_len = len(valid_data)
                output = model(valid_data).squeeze(2).cpu()  # Adjusted to pass only valid_data
                # Proceed with your RMSE calculation

                
                
                for j in range(data_len):
                    if j < seq_len-1:
                    
                        pred_sum[:j+1] += output[j, -(j+1):]
                        pred_cnt[:j+1] += 1
                    elif j <= data_len-seq_len:
                        pred_sum[j-seq_len+1:j+1] += output[j]
                        pred_cnt[j-seq_len+1:j+1] += 1
                    else:
                        pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += output[j, :(data_len-j)]
                        pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += 1
                truth = torch.tensor([valid_label[j,-1] for j in range(len(valid_label)-seq_len+1)], dtype=torch.float) * Rc
                pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
                pred = pred_sum/pred_cnt * Rc
                
                
                mse = float(torch.sum(torch.pow(pred-truth, 2)))
                rmse = math.sqrt(mse/data_len)
                if score:
                    sc = get_score(pred, truth)
                    total_score += sc
                    if testing:
                        print(f'RMSE for {i} is {rmse}, score is {sc}')
                        print('-'*20)
                        
                else:
                    if testing:
                    
                        print(f'RMSE for {i} is {rmse}')
                        print('-'*20)
                        
                total_loss += rmse
        if score:
            return total_score/len(val_data), total_loss/len(val_data)
        else:        
            return total_loss/len(val_data)
                
               

    


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Rc = 130
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=70)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_S4layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='FD001',help="decide source file", choices=['FD001','FD002','FD003','FD004'])
    args = parser.parse_args()
    print(f"Running with args: {args}")
   
    
    exp_name = "s4rul + Attention"
    
    

    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    
    d_input = 24  
    Rc = 130
    
    seq_len = args.seq_len
    dataset = args.dataset
    
    model = S4ModelForRUL(d_input=d_input, d_model=256, n_layers=1, dropout=0.1, max_len=seq_len)
    model.to(device)
    Loss = nn.MSELoss()
    Loss.to(device)


    epochs = args.epochs
    opt, sch = setup_optimizer(
        model, lr=0.02, weight_decay=1e-3, epochs=epochs
        )
    
   
    # TODO CHANGE DATA LOADING
    tr = np.loadtxt("save/"+dataset+"/train"+dataset+".txt", dtype=str).tolist()
    val = np.loadtxt("save/"+dataset+"/valid"+dataset+".txt", dtype=str).tolist()
    ts = np.loadtxt("save/"+dataset+"/test"+dataset+".txt", dtype=str).tolist()
    
    all = tr+val+ts
    
    tr = random.sample(all, int(len(all)*0.7))
    val = random.sample(list(set(all)-set(tr)), int(len(all)*0.12))
    ts = list(set(all)-set(tr)-set(val))
    
    if args.train:
        if args.wandb:
            key = np.loadtxt("key.txt", dtype=str).tolist()
            wandb.login(key = key)

            wandb.init(project="S4RUL", name=exp_name)
        print("="*20)
        print("Training")
        print("="*20)
        data = os.listdir("data/units")
        min_rmse = train(tr, model, Loss, opt, seq_len, epochs, device, exp_name, val)
        print(f"Minimum RMSE: {min_rmse}")
    
    print("="*20)
    print("Testing")
    print("="*20)
    print(validate(model, seq_len, device, ts, testing=True, score=True))
    