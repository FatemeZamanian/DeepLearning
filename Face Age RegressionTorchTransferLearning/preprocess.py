

#download dataset:

# import os
# import shutil
# import kaggle

# try:
#     os.makedirs('.kaggle')
# except:
#     print('File exist')

# shutil.copy('kaggle.json','.kaggle')


# kaggle.api.authenticate()

# kaggle.api.dataset_download_files('dataset', path='jangedoo/utkface-new', unzip=True)

import os
import cv2
import pandas as pd

w=h=224
batch_size=32

images=[]
labels=[]
for image_name in os.listdir('crop_part1')[0:7000]:
  parts=image_name.split('_')
  labels.append(int(parts[0]))
  img=cv2.imread(f'crop_part1/{image_name}')
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  images.append(img)

images=pd.Series(images,name='images')
ages=pd.Series(labels,name='ages')
df=pd.concat([images,ages],axis=1)

under_4=[]
for i in range(len(df)):
  if df['ages'].iloc[i]<=4:
    under_4.append(df.iloc[i])

under_4=pd.DataFrame(under_4)
under_4=under_4.sample(frac=0.3)

up_4=df[df['ages']>4]
df=pd.concat([under_4,up_4])
df=df[df['ages']<90]


os.makedirs('new')

df=pd.DataFrame(df)
new_df=pd.DataFrame(columns=['image','age'])
for i in range(len(df)):
  new_df.loc[i]={'image':f'{i}.jpg','age':df['ages'].iloc[i]}
  df['images'].iloc[i]=cv2.resize(df['images'].iloc[i],(w,h))
  cv2.imwrite(f'new/{i}.jpg',df['images'].iloc[i])

import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from torch.utils.data import Dataset
from PIL import Image

class PathologyPlantsDataset(Dataset):
  """
  The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
  """
  def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
    
  def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
  def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, -1]
        
        if self.transform:
            image = self.transform(image)
    
        return image, label


transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0), (1))
])


dataset= PathologyPlantsDataset(data_frame=new_df,root_dir='new', transform=transform)
train_data_size=int(len(dataset)*0.9)
train_set, test_set = torch.utils.data.random_split(dataset, [train_data_size, len(dataset)-train_data_size])