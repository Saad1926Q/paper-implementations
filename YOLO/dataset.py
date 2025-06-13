import torch
from torch.utils.data import Dataset
import random

class DummyYOLODataset(Dataset):
    def __init__(self, size=100, image_shape=(3, 448, 448), S=7, B=2, C=20):
        self.size = size
        self.image_shape = image_shape
        self.S = S
        self.B = B
        self.C = C
        self.target_dim = B * 5 + C  
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        image = torch.randn(self.image_shape)
        target = torch.zeros((self.S, self.S, self.target_dim))
        
        num_objects = random.randint(1, 3) 

        for n in range(num_objects):
            # Choosing random grid cell
            i=random.randint(0,self.S-1)
            j=random.randint(0,self.S-1)


            x=random.random()  # bbox x coordinate bw 0 and 1
            y=random.random() # bbox y coordinate bw 0 and 1
            w=random.random() # normalized bbox width
            h=random.random() # normalized bbox height

            target[i, j, 0:5] = torch.tensor([x, y, w, h, 1.0])  # the last one means confidence = 1

            class_idx = random.randint(0,self.C-1)  #Choosing a random class

            target[i, j, self.B * 5 + class_idx] = 1.0  # So we are saying probability for this class is 1 whereas for others it is 0

        return image,target