import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from paddle.io import Dataset
import paddle
from paddle.vision.transforms import *

train_path = 'work/datas/Butterfly20/Butterfly20'
target_path = ''
label_path = 'work/datas/Butterfly20/data_list.txt'
test_data_dir='work/datas/Butterfly20_test'
test_path='work/datas/Butterfly20_test/testpath.txt'
spicies_path = 'work/datas/Butterfly20/species.txt'

#数据预处理
class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        self.data_list = []
        with open(label_path,encoding='utf-8') as f:
            for line in f.readlines():
                image_path,genus_label,species_label = line.strip('\n').split(' ')
                self.data_list.append([image_path,species_label])
        # 2. 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image_path, label = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image.astype('float32')
        # 3. 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        label = int(label)-1
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

# 数据预处理和数据增强
transform_train = Compose([
    RandomRotation(40),
    RandomHorizontalFlip(0.4),
    RandomVerticalFlip(0.1),
    Resize(size=(224, 224)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5],data_format='HWC'),
    Transpose()])

#数据加载
train_dataset = MyDataset(label_path, transform_train)
print(train_dataset[0].__getitem__(0).shape)
print(train_dataset.__len__())

#模型组网
res50 = paddle.vision.models.resnet50(num_classes=20)
paddle.summary(res50,(1,3,224,224))

#封装模型
model = paddle.Model(res50)
#调参
#paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
#paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=paddle.nn.CrossEntropyLoss(), 
              metrics=paddle.metric.Accuracy())

#训练
model.fit(train_dataset, 
          epochs=256,
          batch_size=32,
          verbose=1)

# 用 evaluate 在训练集上对模型进行验证
eval_result = model.evaluate(train_dataset, verbose=1)
print(eval_result)

class InferDataset(Dataset):
    def __init__(self, data_dir, image_paths, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本映射到列表中
        """
        super(InferDataset, self).__init__()
        self.data_list = []
        with open(image_paths,encoding='utf-8') as f:
            for line in f.readlines():
                image_path = test_data_dir+'/'+line.strip('\n')
                self.data_list.append(image_path)
        # 2. 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image.astype('float32')
        # 3. 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

transform_test = Compose([
    Resize(size=(224,224)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5],data_format='HWC'),
    Transpose()])
#加载测试集
test_dataset= InferDataset(test_data_dir,test_path,transform_test)
print(test_dataset.__getitem__(0).shape)
print(test_dataset.__len__())

test_result = model.predict(test_dataset)

species_dict={}
with open(spicies_path) as f:
    for line in f:
        a,b = line.strip("\n").split(" ")
        species_dict[int(a)-1]=b

print(species_dict)

with open('model_result.txt','w')as f:
    for i in range(0,200):
        f.write(species_dict[test_result[0][i].argmax()]+'\n')