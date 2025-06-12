import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.nn import Sequential,Conv2d,LeakyReLU,Linear,MaxPool2d,AdaptiveAvgPool2d,Flatten,Dropout
import torch.nn.functional as F

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
        
    

yolo_pretrain=YOLOPretrain()

for images,labels in pretraining_loader:
    print(yolo_pretrain(images))
    print(yolo_pretrain(images).shape)
    break

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Pretrain model here(on ImageNet dataset)


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
    