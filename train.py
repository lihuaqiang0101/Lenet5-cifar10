import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
 
train_number = 20
test_number = 20
batch_size = 8
 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
 
trainsets = torchvision.datasets.CIFAR10(root='./',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainsets,shuffle=True,batch_size=batch_size)
testsets = torchvision.datasets.CIFAR10(root='./',train=False,download=False,transform=transform)
testloader = torch.utils.data.DataLoader(testsets,shuffle=False,batch_size=batch_size)
 
classes = ('飞机', '汽车', '鸟', '猫',
           '鹿', '狗', '青蛙', '马', '船', '卡车')
classes_english = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16,120,5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape([-1,120])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
 
def imgshow(img,target,predict,acc):
    #对图片作归一化
    img = img/2+0.5
    plt.ion()
    img_to_numpy = img.numpy()
    plt.imshow(np.transpose(img_to_numpy,(1,2,0)))
    plt.text(-30,55,'Actual classes:',fontsize=15)
    plt.text(-30, 85, 'Predict classes:', fontsize=15)
    for i in range(8):
        plt.text(35*i,70,'{}'.format(classes_english[target[i]]),fontsize=15)
        plt.text(35 * i, 100, '{}'.format(classes_english[predict[i]]), fontsize=15)
    plt.text(-30, 115, 'Accuracy rate:{}'.format(acc), fontsize=15)
    plt.pause(1.5)
    plt.clf()
    # plt.waitforbuttonpress(0)
 
if __name__ == '__main__':
    net = LeNet_5()
    LOSS = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.001,params=net.parameters())
    for i in range(train_number):
        Loss = 0
        num = 0
        for index,data in enumerate(trainloader):
            print('*'*30)
            x,target = data
            actual_class = []
            predict_class = []
            for classindex in target.numpy():
                actual_class.append(classes[classindex])
            print('真实类别：',actual_class)
            predict = net(x)
            for index in predict:
                index = index.detach().numpy()
                index = np.argmax(index)
                predict_class.append(classes[index])
            print('预测类别：',predict_class)
            optimizer.zero_grad()
            loss = LOSS(predict,target)
            Loss = Loss+loss
            loss.backward()
            optimizer.step()
            num = num+1
            print('*'*30)
        Loss = Loss/num
        print('训练第{0}次，损失为：{1}'.format(i+1,loss.item()))
    #训练完以后开始做测试
    dataiter = iter(testloader)#生成一个可迭代对象
    for _ in range(test_number):
        n = 0
        imgs,lable = dataiter.next()#迭代一次
        img = torchvision.utils.make_grid(imgs)
        outputs = net(imgs)
        Predict = []
        for Index in outputs:
            Index = Index.detach().numpy()
            Index = np.argmax(Index)
            Predict.append(Index)
        for i in range(8):
            print(classes[lable[i]],'==>',classes[Predict[i]])
        for j in range(8):
            if Predict[j] == lable[j]:
                n += 1
        imgshow(img, lable,Predict,n/batch_size)
    torch.save(net, 'net.pkl')  # save entire net
    torch.save(net.state_dict(), 'net_params.pkl')  # save parameters
