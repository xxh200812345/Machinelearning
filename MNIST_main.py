#coding=utf-8
import torch #导入pytorch模块
import torchvision #导入torchvision模块，其中包括一些计算机视觉任务的数据集、模型等
from torch.utils.data import DataLoader #导入DataLoader类
import torch.nn as nn #导入nn模块
import torch.nn.functional as F #导入torch.nn.functional模块
import torch.optim as optim #导入optim模块
import matplotlib.pyplot as plt #导入matplotlib.pyplot模块用于可视化

n_epochs = 3 #训练次数
batch_size_train = 64 #训练集batch_size
batch_size_test = 1000 #测试集batch_size
learning_rate = 0.01 #学习率
momentum = 0.5 #动量
log_interval = 10 #每隔多少个batch输出一次信息
random_seed = 1 #随机数种子
torch.manual_seed(random_seed) #设置随机数种子

#加载MNIST数据集
#训练数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(), #将数据转换为Tensor
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)) #对数据进行标准化
                               ])),
    batch_size=batch_size_train, shuffle=True) #设置batch_size和随机打乱数据集
#测试数据
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)#枚举测试集中的样本
batch_idx, (example_data, example_targets) = next(examples)#获取测试集中的第一个批次数据

# 可视化测试集中的前6个样本
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#定义一个卷积神经网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #输入通道数为1，输出通道数为10，卷积核大小为5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #输入通道数为10，输出通道数为20，卷积核大小为5
        self.conv2_drop = nn.Dropout2d() #2D Dropout层，以一定的概率随机将输入设置为0
        self.fc1 = nn.Linear(320, 50) #输入大小为320，输出大小为50的全连接层
        self.fc2 = nn.Linear(50, 10) #输入大小为50，输出大小为10的全连接层
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #第一个卷积层后接ReLU激活函数和最大池化层
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #第二个卷积层后接2D Dropout层、ReLU激活函数和最大池化层
        x = x.view(-1, 320) #将输出结果展平为一维向量
        x = F.relu(self.fc1(x)) #第一个全连接层后接ReLU激活函数
        x = F.dropout(x, training=self.training) #使用Dropout防止过拟合
        x = self.fc2(x) #第二个全连接层
        return F.log_softmax(x, dim=1) #使用Log Softmax作为输出层的激活函数

 
network = Net() #创建神经网络实例
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum) #定义优化器

train_losses = [] #记录训练损失
train_counter = [] #记录训练次数
test_losses = [] #记录测试损失
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)] #记录测试次数
 
def train(epoch):
    network.train() #将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        #卷积神经网络的梯度是指损失函数对于网络中每个参数的偏导数，它们的值可以用来更新网络参数，使得损失函数的值最小化
        optimizer.zero_grad()#清空梯度
        output = network(data)#将数据输入到神经网络中进行前向传播
        loss = F.nll_loss(output, target) #计算损失
        loss.backward()#计算梯度
        optimizer.step() #更新参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item())) #输出训练日志
            train_losses.append(loss.item())#记录训练
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
 
def test():
    network.eval()# 设置网络为评估模式，该模式下不会计算梯度，减少内存消耗
    test_loss = 0
    correct = 0
    with torch.no_grad():# 关闭梯度计算，加快推理速度
        for data, target in test_loader:
            output = network(data)# 关闭梯度计算，加快推理速度
            test_loss += F.nll_loss(output, target, reduction='sum').item()# 计算测试集上的损失
            pred = output.data.max(1, keepdim=True)[1]# 找到输出中概率最大的类别作为预测值
            correct += pred.eq(target.data.view_as(pred)).sum()# 找到输出中概率最大的类别作为预测值
    test_loss /= len(test_loader.dataset)# 计算测试集上的平均损失
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
 
train(1)#第一次训练模型
 
test()  # 不加这个，后面画图就会报错：x and y must be the same size
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()# 每轮训练完后，在测试集上测试模型性能，并保存测试损失
 
 
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue') #绘制训练损失图表
plt.scatter(test_counter, test_losses, color='red') #绘制测试损失图表
plt.legend(['Train Loss', 'Test Loss'], loc='upper right') #添加图例
plt.xlabel('number of training examples seen') #x轴标签
plt.ylabel('negative log likelihood loss') #y轴标签

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = network(example_data)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1) #将图表分成2行3列，每个子图占一个位置
    plt.tight_layout() #调整子图的布局
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none') #绘制灰度图像
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item())) #添加标题
    plt.xticks([]) #取消x轴刻度
    plt.yticks([]) #取消y轴刻度
plt.show() #显示图表

 
# ----------------------------------------------------------- #
# 首先创建了一个新的Net模型continued_network和一个新的优化器continued_optimizer，
# 然后使用torch.load()函数分别加载之前保存好的模型和优化器状态字典（即模型和优化器的参数）。
# 这里假设之前保存好的模型状态字典文件名为model.pth，优化器状态字典文件名为optimizer.pth。
# 然后调用load_state_dict()方法将之前保存的状态字典加载到新创建的模型和优化器中，这样新的模型和优化器就有了之前训练时的状态。

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
 
# 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# 不然报错：x and y must be the same size
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
for i in range(4, 9):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()

# 这段代码用于绘制损失函数的训练和测试曲线
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
