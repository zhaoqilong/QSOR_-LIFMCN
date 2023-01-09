# torch
import torch
from PIL import Image
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.nn.functional import normalize
import csv
from model_metrics import (
    binary_accuracy,
    binary_precison_recall,
    troch_pr_auc,
    binary_auc,
    precision_recall_fscore_k,
    auROC,
)
from ExternalAttention import ExternalAttention
import LICNN_core
import utils
# from LICNN_core import LICNN
import torch.nn.functional as F
from torch.nn import init
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn as sns
# from model.fpn.resnet import resnet
import warnings

warnings.filterwarnings('ignore')


# classes = ['human', 'beach', 'architecture', 'car', 'dinosaur', 'elephent', 'flower', 'horse', 'mountain',
#            'food']

class DataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.batch_size = batch_size
        self.file_names = os.listdir(path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        pic_index = self.file_names[index].split('.')[0]
        pic_index = int(pic_index)

        label = 0
        with open("train.csv") as f:
            reader = csv.reader(f)
            # print("index ",index)
            # print("pic_index",pic_index)
            for row in reader:
                # first_row=next(reader)
                if int(pic_index) == int(row[0]):
                    # print(row[1])
                    label = torch.Tensor([int(row[1])]).long()

        im = Image.open(self.path + '/%d.png' % pic_index)
        H, W = im.size
        if H != 256:
            im = im.rotate(-90, expand=True)
        im = torchvision.transforms.functional.resize(im, size=(128, 128))  # (96, 64)
        im = torchvision.transforms.functional.to_tensor(im)  # [3,96,64] [3,96,96] [3,128,128]
        # print("aaa",aaa)
        if aaa == 0:
            img = utils.load_image(self.path + '/%d.png' % pic_index)
            # print(self.path + '/%d.png' % pic_index)
            _, attention_map, _ = LICNN_core.LICNN.create_saliency_map(LICNN_core.to_tensor(img))
            # net = LICNN_core.LICNN()
            # _, attention_map = net(LICNN_core.to_tensor(img), given_id=int(label))
            # print("top100 ",top100)
            # print("attention_map ",attention_map.shape[0])
            attention = np.random.rand(128, 128)
            for i in range(128):
                for j in range(128):
                    attention[i][j] = attention_map[i + 48][j + 48]
            # print(attention.shape)
            im = (im * attention).type(torch.FloatTensor)
            # print(self.classes[int(pic_index / 100)])
            # im.show()  # 查看图片

        # print(im.shape)
        return im, label


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=105):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.att1 = ExternalAttention(d_model=128, S=8) #[batch_size,32,128,128]
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.att2 = ExternalAttention(d_model=64, S=8) #[batch_size, 64, 64, 64]
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.att3 = ExternalAttention(d_model=32, S=8) #[batch_size, 128, 32, 32]
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.att4 = ExternalAttention(d_model=16, S=8) #[batch_size, 256, 16, 16]
        self.linear = nn.Linear(256*16 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # out = self.att1(out) #good!
        # print(out.shape)
        out = self.layer2(out)
        # out = self.att2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape) #[64, 128, 32, 32]
        out = self.att3(out) #good!!!
        # print(out.shape)
        # print(out.shape)
        out = self.layer4(out)
        # out = self.att4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3]) #good!!!


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

"""
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        self.att1 = ExternalAttention(d_model=64, S=8)
        # [batch_size,16,48,32] [batch_size,16,48,48] [batch_size,16,64,64]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.att2 = ExternalAttention(d_model=32, S=8)
        # [batch_size,32,24,16] [batch_size,32,24,24] [batch_size,32,32,32]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att3 = ExternalAttention(d_model=16, S=8)
        # [batch_size,64,12,8] [batch_size,64,12,12] [batch_size,64,16,16]
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(16, 16))  # (12, 8)
        self.relu4 = nn.ReLU()
        # [batch_size,128,1,1] [batch_size,128,1,1]
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        # self.relu5 = nn.ReLU()
        # [batch_size,64,1,1]
        self.linear1 = nn.Linear(128, 105)
        self.net = nn.Sequential(self.conv1, self.relu1, self.max_pool1, self.conv2, self.relu2, self.max_pool2,
                                 self.conv3, self.relu3, self.max_pool3, self.conv4, self.relu4)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        input = self.conv1(input)
        input = self.relu1(input)
        input = self.max_pool1(input)
        # ea = ExternalAttention(d_model=input.shape[3], S=8)
        input = self.att1(input)
        # print(input.shape)
        input = self.conv2(input)
        input = self.relu2(input)
        input = self.max_pool2(input)
        input = self.att2(input)
        # print(input.shape)
        # input = F.dropout(input, 0.2, training=self.training)
        input = self.conv3(input)
        input = self.relu3(input)
        input = self.max_pool3(input)
        input = self.att3(input)
        # print(input.shape)
        input = self.conv4(input)
        input = self.relu4(input)
        # input = self.conv5(input)
        # output = self.relu5(input)
        # print(input.shape)
        output = self.linear1(input.view(-1, 128))
        # print(output.shape)
        output = torch.sigmoid(output)
        output = torch.softmax(output, dim=1)
        return output
"""


class Trainer(object):
    def __init__(self, train_dataloader, test_dataloader, model, epoch=20000, lr=0.0001):  # epoch=20000
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.cross_entropy = nn.CrossEntropyLoss()

    def start(self):
        self.epoch_losses = []
        self.epoch_test_losses = []
        self.epoch_accs = []
        self.epoch_test_accs = []
        for i in range(1, self.epoch + 1):
            epoch_loss = 0
            epoch_test_loss = 0
            epoch_acc = 0
            epoch_test_acc = 0
            epoch_precision = 0
            epoch_recall = 0
            epoch_f1 = 0
            epoch_auroc = 0
            p, r, f, a = 0, 0, 0, 0
            # 训练一个epoch
            # print("176")
            for batch_data in self.train_dataloader:
                # 训练一个batch
                im, label = batch_data
                # print("179")
                im, label = im.to(self.device), label.to(self.device).view(-1)
                result = self.train_batch(im, label)
                epoch_loss += result[0]
                predict_output = result[1]
                epoch_acc += self.get_acc(predict_output=predict_output, label=label)
                p, r, f, a = self.get_other_metrics(predict_output=predict_output, label=label)
                # epoch_precision+=p
                epoch_recall += r
                # epoch_f1+=f
                epoch_auroc += a
            # print("191")
            epoch_loss /= len(self.train_dataloader)
            epoch_acc /= len(self.train_dataloader)
            # epoch_precision /= len(self.train_dataloader)
            epoch_recall /= len(self.train_dataloader)
            # epoch_f1 /= len(self.train_dataloader)
            epoch_f1 = 2 * (epoch_acc * epoch_recall) / (epoch_acc + epoch_recall)
            epoch_auroc /= len(self.train_dataloader)
            self.epoch_losses.append(epoch_loss)
            self.epoch_accs.append(epoch_acc)
            # 每训练一个epoch，测试一次
            epoch_test_loss, epoch_test_acc, epoch_test_recall, epoch_test_f1, epoch_test_auroc = self.test(
                save_result=(i == self.epoch))
            self.epoch_test_losses.append(epoch_test_loss)
            self.epoch_test_accs.append(epoch_test_acc)
            # print("epoch %d:train loss %f,test loss %f,train acc %f,test acc %f, recall %f, f1score %f, auroc %f" % (
            #     i, epoch_loss, epoch_test_loss, epoch_acc, epoch_test_acc, epoch_recall,epoch_f1,epoch_auroc))
            print(
                "epoch %d: loss %f, acc %f, recall %f, f1score %f, auroc %f; valid loss %f, acc %f, recall %f, f1score %f, auroc %f" % (
                    i, epoch_loss, epoch_acc, epoch_recall, epoch_f1, epoch_auroc, epoch_test_loss, epoch_test_acc,
                    epoch_test_recall, epoch_test_f1, epoch_test_auroc))

    def get_acc(self, predict_output, label):
        # acc = None
        # k = 10
        with torch.no_grad():
            # test_num = label.size()[0]
            # max_index = predict_output.max(dim=1)[1]
            # count = (max_index == label).sum().item()
            # return count / test_num

            # value, index=predict_output.topk(k,dim=1,largest=True, sorted=True)
            # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # n_y=torch.Tensor(list(label.shape)[0],k).copy_(torch.unsqueeze(label,dim=1)).to(device)
            # correct=torch.sum((index==n_y).float(),dim=0)
            # acc=correct.sum()/len(correct)
            # return acc

            # print(predict_output)
            # print(label)

            epoch_topk_acc = 0
            epoch = 0
            topk = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            maxk = max(topk)
            batch_size = label.size(0)
            value, predict = predict_output.topk(10, 1, True, True)
            # print("value ", value)
            # print("predict ", predict)
            pred = predict.t()
            correct = pred.eq(label.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))
                epoch += 1
            # print("epoch ",epoch)
            # print("len(res) ",len(res))
            epoch_topk_acc += res[9].item()
            # topk_acc=epoch_topk_acc / len(res)
            return epoch_topk_acc

    def get_other_metrics(self, predict_output, label):
        with torch.no_grad():
            _, predict = predict_output.topk(10, 1, True, True)
            precision, recall, f1score = precision_recall_fscore_k(label.cpu().numpy().tolist(),
                                                                   predict.cpu().numpy().tolist(), num=10)
            # print(precision, recall, f1score)
            # f1score = 2 * (topk_acc * recall / (topk_acc + recall))
            # print(labels.cpu().numpy())
            # print(predict.cpu().numpy())
            _, predict = predict_output.topk(20, 1, True, True)  # 202020202020202020202020
            auroc = auROC(label.cpu().numpy().tolist(), predict.cpu().numpy().tolist())
            return precision, recall, f1score, auroc

    def train_batch(self, img, label):
        self.optimizer.zero_grad()
        # 训练一个batch
        predict_output = self.model(img)  # +
        # print(predict_output.size())
        batch_loss = self.cross_entropy(predict_output, label)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item(), predict_output
        # epoch_acc += self.get_acc(predict_output=predict_output, label=label)

    def test(self, save_result=False):
        epoch_test_loss = 0
        epoch_test_acc = 0
        predict_output = None
        epoch_test_precision = 0
        epoch_test_recall = 0
        epoch_test_f1 = 0
        epoch_test_auroc = 0
        p, r, f, a = 0, 0, 0, 0
        with torch.no_grad():
            for img, label in self.test_dataloader:
                img, label = img.to(self.device), label.to(self.device).view(-1)
                predict_output = self.model(img)
                batch_loss = self.cross_entropy(predict_output, label)
                epoch_test_loss += batch_loss.item()
                epoch_test_acc += self.get_acc(predict_output=predict_output, label=label)
                p, r, f, a = self.get_other_metrics(predict_output=predict_output, label=label)
                # epoch_test_precision+=p
                epoch_test_recall += r
                # epoch_f1+=f
                epoch_test_auroc += a
                # print("191")
            epoch_test_loss /= len(self.test_dataloader)
            epoch_test_acc /= len(self.test_dataloader)
            # epoch_precision /= len(self.train_dataloader)
            epoch_test_recall /= len(self.test_dataloader)
            # epoch_f1 /= len(self.train_dataloader)
            epoch_test_f1 = 2 * (epoch_test_acc * epoch_test_recall) / (epoch_test_acc + epoch_test_recall)
            epoch_test_auroc /= len(self.test_dataloader)
        if save_result:
            self.predict_output = predict_output
        return epoch_test_loss, epoch_test_acc, epoch_test_recall, epoch_test_f1, epoch_test_auroc


"""
    def draw(self):
        # 得到预测结果
        max_index = self.predict_output.max(dim=1)[1]
        # print(max_index[1])
        test_num = max_index.size()[0]
        # print(test_num)
        # [200,1]
        label = [item for item in self.test_dataloader][0][1]
        # print(label)
        # [200,1]
        matrix = torch.tensor((), dtype=torch.float64)
        matrix = matrix.new_zeros((105, 105))
        # matrix=torch.tensor(np.random.rand(105,105), dtype=torch.float64)
        # 统计
        for i in range(test_num):
            print(label[i][0].tolist(), max_index[i].tolist())
            matrix[label[i][0].tolist(), max_index[i].tolist()] += 1
        # 绘制混淆矩阵，纵坐标真实Label，横坐标预测Label
        print(matrix)
        matrix = normalize(matrix,dim=1).view(105,105).numpy()
        print(matrix)
        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        # 设置坐标轴刻度数量
        ax.set_xticks(np.arange(105))
        ax.set_yticks(np.arange(105))
        #设置坐标轴刻度名字
        ax.set_xticklabels(np.arange(1,106))
        ax.set_yticklabels(np.arange(1,106))
        # ax.tick_params(axis='both', which='minor', labelsize=1)
        # plt.xticks(fontsize=5)
        # plt.yticks(fontsize=5)
        # # 设置坐标标签字体大小
        # ax.set_xlabel('region',fontsize=5)
        # ax.set_ylabel('kind',fontsize=5)
        # 横坐标刻度名字倾斜45度
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")
        for i in range(105):
            for j in range(105):
                text = ax.text(j, i, str(matrix[i, j])[0:4], ha="center", va="center", color="w")
        ax.set_title("matrix")
        fig.tight_layout()
        fig.colorbar(im, ax=ax)
        plt.savefig('result/matrix.jpg')
"""

if __name__ == "__main__":
    # 准备数据
    aaa = 0
    batch_size = 64  # 800
    train_data = DataSet(path='2d_images')
    test_data = DataSet(path='2d_images_test')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   pin_memory=True)  # pin_memory=True
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True,
                                                  pin_memory=True)  # pin_memory=True

    # 准备模型
    # model = VGG()
    model = ResNet34()
    # y = model(torch.randn(1, 3, 128, 128))  # (1,3,32,32)
    # print(y.size())
    print("有%d个参数" % sum(p.numel() for p in model.parameters()))
    # 开始训练
    trainer = Trainer(train_dataloader=train_dataloader, test_dataloader=test_dataloader, model=model, epoch=30,
                      lr=0.001)  # epoch=1000
    aaa = 1
    # print("351")
    trainer.start()
    # print("353")
    # trainer.draw()

