import torch
import random
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from dataset import DummyYOLODataset
from torch.nn import Sequential,Conv2d,LeakyReLU,Linear,MaxPool2d,AdaptiveAvgPool2d,Flatten,Dropout
from bbox_utils import xywh_to_xyxy,iou

pretraining_transforms=transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Yes I am aware that this is supposed to be ImageNet dataset(but yeah its around 190 GB so gl with that)

pretraining_dataset=datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=1000, transform=pretraining_transforms)

pretraining_validation_dataset = datasets.FakeData(size=20,image_size=(3, 224, 224),num_classes=1000,transform=pretraining_transforms)

pretraining_loader = DataLoader(pretraining_dataset, batch_size=32, shuffle=True)
pretraining_validation_loader = DataLoader(pretraining_validation_dataset, batch_size=32, shuffle=False)

for images,labels in pretraining_loader:
    print(images.shape)
    print(labels.shape)
    break


class YOLOPretrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor=Sequential(
            Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2),
            LeakyReLU(negative_slope=0.1),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
            LeakyReLU(negative_slope=0.1),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(in_channels=192,out_channels=128,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=256,out_channels=256,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),#6
            LeakyReLU(negative_slope=0.1),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),#14
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=512,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),#16
            LeakyReLU(negative_slope=0.1),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            LeakyReLU(negative_slope=0.1),
            Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),#20
            LeakyReLU(negative_slope=0.1),
        )

        self.avgpool = AdaptiveAvgPool2d((1, 1))

        self.classifer=Linear(in_features=1024,out_features=1000)

    def forward(self,X):
        output=self.feature_extractor(X)
        output = self.avgpool(output)
        output=torch.flatten(output,start_dim=1)
        output=self.classifer(output)
        return output

    def compute_accuracy(self,data_loader,device):
        correct_preds=0
        total_preds=0

        for batch_idx,(images,labels) in enumerate(data_loader):
            images=images.to(device)
            labels=labels.to(device)
            logits=self.forward(images)
            probs=F.softmax(logits,dim=1)
            preds=torch.argmax(probs,dim=1)
            correct_preds+=torch.sum(preds==labels)
            total_preds+=labels.size(0)

        accuracy=correct_preds/total_preds
        return accuracy*100

    def pretrain(self,num_epochs,train_loader,validation_loader,device):
        self.to(device)
        optimizer=torch.optim.SGD(self.parameters())

        for epoch in range(num_epochs):
            self.train()

            for batch_idx,(images,labels) in enumerate(train_loader):
                images=images.to(device)
                labels=labels.to(device)
                logits=self.forward(images)
                loss=F.cross_entropy(logits,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                train_acc=self.compute_accuracy(train_loader,device)
                valid_acc=self.compute_accuracy(validation_loader,device)
                print(f"Epoch: {epoch+1}/{num_epochs}, Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {valid_acc:.2f}%")


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')       

yolo_pretrain=YOLOPretrain()
yolo_pretrain.to(device)


for images,labels in pretraining_loader:
    images=images.to(device)
    labels=labels.to(device)
    print(yolo_pretrain(images))
    print(yolo_pretrain(images).shape)
    break




# Pretrain model here(on ImageNet dataset)


dummy_dataset = DummyYOLODataset(size=500)
val_dataset = DummyYOLODataset()

train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


for imgs, tgts in train_loader:
    print(imgs.shape)   # [32, 3, 448, 448]
    print(tgts.shape)   # [32, 7, 7, 30]
    break  


class YOLO(torch.nn.Module):
    def __init__(self,pretrained_network):
        super().__init__()

        self.model=Sequential(
            pretrained_network,
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            LeakyReLU(0.1),
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=2,padding=1),
            LeakyReLU(0.1),
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            LeakyReLU(0.1),
            Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            LeakyReLU(0.1),
            torch.nn.Flatten(),
            Linear(50176, 4096),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(4096, 1470)
        )

    def forward(self,X):
        output=self.model(X)
        output=torch.reshape(output,(-1,7, 7, 30))
        return output
    
    def loss(self,predictions,targets,l_coord=5,l_noobj=0.5):
        total_loss=0


        for example in range(len(predictions)): # Looping through the training examples one at a time
            for grid_x in range(7):
                for grid_y in range(7):
                    # bbox1_x,bbox1_y,bbox1_w,bbox1_h,bbox1_confidence
                    coordinates_error=0
                    class_probability_error=0
                    no_obj_confidence_error=0
                    obj_confidence_error=0

                    target_cell=targets[example,grid_x,grid_y]
                    preds_cell=predictions[example,grid_x,grid_y]


                    confidence=target_cell[4]

                    if confidence==1:
                        pred_bbox1=preds_cell[0:5]
                        pred_bbox2=preds_cell[5:10]

                        ground_truth_box=target_cell[0:5]

                        if iou(pred_bbox1,ground_truth_box)>iou(pred_bbox2,ground_truth_box):
                            responsible_bbox=pred_bbox1
                        else:
                            responsible_bbox=pred_bbox2

                        pred_bbox_x,pred_bbox_y,pred_bbox_w,pred_bbox_h,pred_bbox_confidence=responsible_bbox
                        ground_truth_bbox_x,ground_truth_bbox_y,ground_truth_bbox_w,ground_truth_bbox_h,ground_truth_bbox_confidence=ground_truth_box

                        coordinates_error+=(ground_truth_bbox_x-pred_bbox_x)**2
                        coordinates_error+=(ground_truth_bbox_y-pred_bbox_y)**2
                        coordinates_error+=(torch.sqrt(torch.abs(ground_truth_bbox_w) + 1e-6)-torch.sqrt(torch.abs(pred_bbox_w)+ 1e-6))**2
                        coordinates_error+=(torch.sqrt(torch.abs(ground_truth_bbox_h) + 1e-6)-torch.sqrt(torch.abs(pred_bbox_h) + 1e-6))**2

                        coordinates_error*=l_coord

                        total_loss+=coordinates_error

                        obj_confidence_error+=(pred_bbox_confidence-ground_truth_bbox_confidence)**2

                        total_loss+=obj_confidence_error

                        class_probability_error+=torch.sum((preds_cell[10:]-target_cell[10:])**2)

                        total_loss+=class_probability_error
                    
                    else:
                        no_obj_confidence_error+=(preds_cell[4] - 0) ** 2
                        no_obj_confidence_error+=(preds_cell[9] - 0) ** 2
                        no_obj_confidence_error*=l_noobj
                        total_loss+=no_obj_confidence_error
        
        return total_loss/len(predictions)


    def train_model(self,num_epochs,train_loader,validation_loader,device):
        optimizer=torch.optim.SGD(self.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)

        scheduler =torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,start_factor=0.1,end_factor=1,total_iters=10),
                torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer,factor=1,total_iters=75),
                torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer,factor=0.1,total_iters=30),
                torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer,factor=0.01,total_iters=30)
            ],
            milestones=[10, 85, 115]
        )

        for epoch in range(num_epochs):
            self.train()
            for batch_idx,(images,targets) in enumerate(train_loader):
                images=images.to(device)
                targets=targets.to(device)

                predictions=self.forward(images)
                loss=self.loss(predictions=predictions,targets=targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            with torch.no_grad():
                self.eval()
                val_loss=0
                for val_images, val_targets in validation_loader:
                    val_images = val_images.to(device)
                    val_targets = val_targets.to(device)
                    val_preds = self.forward(val_images)
                    val_loss += self.loss(val_preds, val_targets)
                print(f"validation loss for epoch {epoch} is {val_loss}")

yolo=YOLO(pretrained_network=yolo_pretrain.feature_extractor)
yolo.to(device)

yolo.train_model(1,train_loader=train_loader,validation_loader=val_loader,device=device)
