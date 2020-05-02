from torch.utils.data import Dataset
from PIL import Image

class WeatherDataset(Dataset):
    # define dataset
    def __init__(self,label_list,transforms=None,mode="train"):
        super(WeatherDataset,self).__init__()
        self.label_list = label_list
        self.transforms = transforms
        self.mode = mode
        imgs = []
        if self.mode == "test":
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        if self.mode == "test":
            filename = self.imgs[index]
            img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            return img,filename
        else:
            filename,label = self.imgs[index]
            img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            return img,label


