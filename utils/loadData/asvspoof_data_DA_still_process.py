import numpy as np
import soundfile as sf
import torch,os,librosa
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import lightning as L
from transformers import Wav2Vec2FeatureExtractor
from utils.tools.tools import pad,pad_random    
    
class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                
                # TODO: change the dir to your own data dir
                # label file
                self.train_protocols_file = "/data4/wuyikai/data/spoof24/ASVspoof5.train.metadata.txt"
                self.dev_protocols_file = "/data4/wuyikai/data/spoof24/ASVspoof5.dev.metadata.txt"
                # flac file dir
                self.train_set="/data4/wuyikai/data/spoof24/flac_T/"
                self.dev_set="/data4/wuyikai/data/spoof24/flac_D/"
                # test set 
                # self.eval_protocols = "/data8/wangzhiyong/project/fakeAudioDetection/asvspoof5/asvspoof5/spoof5/ASVspoof5.dev.metadata.txt"
                # self.eval_set = "/data8/wangzhiyong/project/fakeAudioDetection/asvspoof5/asvspoof5/flac_D/"

                self.truncate = args.truncate
                self.predict = args.testset # LA21, DF21, ITW

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                d_label_trn,file_train = genSpoof_list(
                    dir_meta = self.train_protocols_file,
                    is_train=True,is_eval=False)
                
                self.asvspoof5_train_set = Dataset_train(
                    list_IDs = file_train,
                    labels = d_label_trn,
                    base_dir = self.train_set,
                    cut=self.truncate
                    )
   
   
                d_label_dev,file_dev = genSpoof_list(
                    dir_meta =self.dev_protocols_file,
                    is_train=False,is_eval=False)
                
                self.asvspoof5_val_set = Dataset_train(
                    list_IDs = file_dev,
                    labels = d_label_dev,
                    base_dir = self.dev_set,
                    cut=self.truncate
                    )
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                # TODO:
                pass

            if stage == "predict":
                # TODO:
                pass

                    
                    
                

        def train_dataloader(self):
            return DataLoader(self.asvspoof5_train_set, batch_size=self.args.batch_size, shuffle=True,drop_last = True,num_workers=4)

        def val_dataloader(self):
            return DataLoader(self.asvspoof5_val_set, batch_size=self.args.batch_size, shuffle=False,drop_last = False,num_workers=4)            

        def test_dataloader(self):                
            pass

        def predict_dataloader(self):
            pass
      
def norm(X_pad):
    mean_x = X_pad.mean()
    var_x = X_pad.var()
    return np.array([(x - mean_x) / np.sqrt(var_x + 1e-7) for x in X_pad])      
      


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _,key,_,_,_,label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _,key,_,_,_,label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list



class Dataset_train(Dataset):
    def __init__(self,list_IDs, labels, base_dir,cut):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut # take ~4 sec audio 
        self.fs  = 16000
     
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir+key+'.flac', sr=16000)  
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[key] 
        # tensor, label, filename
        return x_inp, target, key



class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir,cut):
        
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir+key+'.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
      
      
      
      
