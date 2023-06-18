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
transform_train = Compose([ColorJitter(0.1, 0.4, 0.4, 0.1),RandomRotation(10), Resize(size=(224, 224)),RandomHorizontalFlip(0.4),
    RandomVerticalFlip(0.1), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),Resize(size=(224,224)),Transpose()])

#数据加载
train_dataset = MyDataset(label_path, transform_train)

print(train_dataset[1136][0].shape)
print(train_dataset[1136][1])