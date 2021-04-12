

# 手动写一个类来读取数据
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import random
# input size
size = 224
# 使用image net的mean std 简单的数据增强
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    normalize
])

val_transformer_ImageNet = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    normalize
])
# 目录文件
data_dir = 'data'
# 为了划分数据集，和自定义transform 所以参考如下链接写了一个这个
# refer https://blog.csdn.net/ncc1995/article/details/91125964
class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

def split_Train_Val_Data(data_dir, ratio, batch_size=4):
    global train_len
    global val_len
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)     # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
#     print(dataset.samples)
    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):   # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        #print(num_sample_train)
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val
        # 这里打乱一下数据，实验表明，不打乱也没事
        random.seed(7)
        random.shuffle(data)

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)

    train_len = len(train_inputs)
    val_len = len(val_inputs)
    # print("train_length:%d,val length:%d" %(train_len,val_len))

    train_dst = MyDataset(train_inputs, train_labels, train_transformer_ImageNet)
    valid_dst = MyDataset(val_inputs, val_labels, val_transformer_ImageNet)
    train_dataloader = DataLoader(train_dst,
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dst,
                                  batch_size=batch_size, shuffle=False)
    data_loader = [train_dataloader, val_dataloader]
    dataloders = {x:  data_loader[i] for i,x in enumerate(['train', 'val']) }
    dataset_sizes = {'train':train_len, 'val':val_len}
    print(dataset_sizes)
    return dataloders,dataset_sizes
# 定义pytorch的dataloader，数据划分0.9
# data_loader = split_Train_Val_Data(data_dir,(0.9,0.1))
# 为了保证后面和官方的baseline一致，所以可以这么写

# use gpu or not
# use_gpu = torch.cuda.is_available()
