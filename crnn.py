import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy
import os
import resnet as res


# 超参数
Epoch = 100
Batch = 8
Time_step = 10     # 考虑n个时间关联序列
class_num = 6      # 种类
LR = 0.00001


# 加载自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.file_name_list = os.listdir(root_dir)
        self.root_dir = root_dir

    def __getitem__(self, idx):
        data = numpy.load(self.root_dir + self.file_name_list[idx])
        label = int(self.file_name_list[idx].replace('.npy', "")) % 10 - 1
        return data, label

    def __len__(self):
        return len(self.file_name_list)


train_dataset = CustomDataset(root_dir='./traindata/')
test_dataset = CustomDataset(root_dir='./testdata/')

train_loader = DataLoader(dataset=train_dataset, batch_size=Batch, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=Batch, shuffle=False)


# 开启GPU，若没有则使用CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# CRNN模型
class CRNN(nn.Module):
    def __init__(self, res_output=32, rnn_ouput=64):
        super(CRNN, self).__init__()
        self.resnet = res.ResNet(output_length=res_output)
        self.lstm = nn.LSTM(
            input_size=res_output,
            hidden_size=rnn_ouput,  # 隐藏神经单元数
            num_layers=1,           # 隐层数目
            batch_first=True
        )
        self.fc = nn.Linear(rnn_ouput, class_num)

    def forward(self, x):
        temp = torch.zeros((Batch, Time_step, 32)).to(device)
        for i in range(Time_step):
            temp[:, i] = self.resnet(x[:, i, :, :, :])  # ResNet
        temp, (h_n, h_c) = self.lstm(temp)
        temp = self.fc(temp[:, -1, :])
        return temp


# 准备
model = CRNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# 更新学习率
def update_lr(optim,lr):
    for param_group in optim.param_groups:
        param_group['lr']=lr


# 训练
totle_loss = 0
cnt = 0
acc = 0
curr_lr = LR
for epoch in range(Epoch):
    for i, (data, label) in enumerate(train_loader):
        data = data.float().to(device).view(Batch, Time_step, -1, 224, 224)  # 增加一个维度
        label = label.to(device)
        outputs = model(data)

        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totle_loss += loss.item()

    # 测试
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.float().to(device)
            label = label.to(device)

            outputs = model(data.view(Batch, Time_step, -1, 224, 224))

            cnt += 1
            lab = label.tolist()
            outputs = outputs.tolist()
            out = []
            for m in range(len(outputs)):
                out.append(outputs[m].index(max(outputs[m])))
            for n in range(Batch):
                if lab[n] == out[n]: acc += 1
            print(lab)
            print(out)
    print(totle_loss/cnt)
    print(acc/(cnt*Batch)*100)

    # 保存模型
    torch.save(model.state_dict(), './model/epoch%d_acc%d.ckpt'
               % (epoch + 1, int(acc/(cnt*Batch)*100)))

    totle_loss = 0
    cnt = 0
    acc = 0

    # 学习率衰减
    if((epoch % 10 + 1) == 0):
        curr_lr = curr_lr * (0.5 ** Epoch)
        update_lr(optimizer, curr_lr)
