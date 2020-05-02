import os
import torch
import warnings
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from glob import glob
from PIL import Image,ImageFile
from config import configs
from models.model import get_model
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from utils.misc import get_files
from IPython import embed

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
len_data = 0

class WeatherTTADataset(Dataset):
    def __init__(self,labels_file,aug):
        imgs = []
        for index, row in labels_file.iterrows():
            imgs.append((row["FileName"],row["type"]))
        self.imgs = imgs
        self.length = len(imgs)
        global len_data
        len_data = self.length
        self.aug = aug
        self.Hflip = transforms.RandomHorizontalFlip(p=1)
        self.Vflip = transforms.RandomVerticalFlip(p=1)
        self.Rotate = transforms.functional.rotate
        self.resize = transforms.Resize((configs.input_size,configs.input_size))
        self.randomCrop = transforms.Compose([transforms.Resize(int(configs.input_size * 1.2)),
                                            transforms.CenterCrop(configs.input_size),
                                            ])
    def __getitem__(self,index):
        filename,label_tmp = self.imgs[index]
        img = Image.open(configs.test_folder + os.sep + filename).convert('RGB')
        img = self.transform_(img,self.aug)
        return img,filename

    def __len__(self):
        return self.length
    def transform_(self,data_torch,aug):
        if aug == 'Ori':
            data_torch = data_torch
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Hflip':
            data_torch = self.Hflip(data_torch)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Vflip':
            data_torch = self.Vflip(data_torch)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Rotate_90':
            data_torch = self.Rotate(data_torch, 90)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Rotate_180':
            data_torch = self.Rotate(data_torch, 180)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Rotate_270':
            data_torch = self.Rotate(data_torch, 270)
            data_torch = self.resize(data_torch)
        if aug == 'Crop':
            # print(data_torch.size)
            data_torch = self.randomCrop(data_torch)
            data_torch = data_torch
        if aug == 'Crop_Hflip':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Hflip(data_torch)
        if aug == 'Crop_Vflip':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Vflip(data_torch)
        if aug == 'Crop_Rotate_90':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Rotate(data_torch, 90)
        if aug == 'Crop_Rotate_180':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Rotate(data_torch, 180)
        if aug == 'Crop_Rotate_270':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Rotate(data_torch, 270)
        data_torch = transforms.ToTensor()(data_torch)
        data_torch = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(data_torch)
        return data_torch

#aug = ['Ori','Ori_Hflip','Ori_Vflip','Ori_Rotate_90','Ori_Rotate_180','Ori_Rotate_270',
     # 'Crop','Crop_Hflip','Crop_Vflip','Crop_Rotate_90','Crop_Rotate_180','Crop_Rotate_270']
aug = ['Ori_Hflip']

cpk_filename = configs.checkpoints + os.sep + configs.model_name + "-checkpoint.pth.tar"
best_cpk = cpk_filename.replace("-checkpoint.pth.tar","-best_model.pth.tar")
checkpoint = torch.load(best_cpk)
cudnn.benchmark = True
model = get_model()
model.load_state_dict(checkpoint['state_dict'])
model.eval()
test_files = pd.read_csv(configs.submit_example)

with torch.no_grad():
    y_pred_prob = torch.FloatTensor([])
    for a in tqdm(aug):
        print(a)
        test_set = WeatherTTADataset(test_files, a)
        test_loader = DataLoader(dataset=test_set, batch_size=configs.bs, shuffle=False,
                                 num_workers=4, pin_memory=True, sampler=None)
        total = 0
        correct = 0
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = outputs[:, :45]
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            # print(outputs.shape)
            y_pred_prob = torch.cat([y_pred_prob, outputs.to("cpu")], dim=0)
    #embed()
    y_pred_prob = y_pred_prob.reshape((len(aug), len_data, configs.num_classes))
    y_pred_prob = torch.sum(y_pred_prob, 0) / (len(aug) * 1.0)
    _, predicted_all = torch.max(y_pred_prob, 1)
    predicted = predicted_all + 1
    test_files.type = predicted.data.cpu().numpy().tolist()
    test_files.to_csv('./submits/%s_baseline.csv' % configs.model_name, index=False)