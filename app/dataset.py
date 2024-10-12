import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

# Define paths
# train_img_dir = '/tcdata/train/img'
# train_label_dir = '/tcdata/train/label'
# train_label_csv = '/tcdata/train/label.csv'
# test_img_dir = '/tcdata/test/img'
# output_dir = './submit/label'
# output_csv = './submit/label.csv'
# submit_zip = './submit.zip'


# 标准化变换
default_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


class TrainImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, label_csv, transform=None):
        """
        Args:
            img_dir (str): 图像文件夹的路径
            label_dir (str): 标签文件夹的路径
            label_csv (str): 包含标签信息的 CSV 文件路径
            transform (callable, optional): 图像的变换/标准化操作
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform is not None else default_transform

        # 读取 CSV 文件，假设它包含 'filename' 和 'label' 两列
        self.label_df = pd.read_csv(label_csv)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        # 获取图像文件名和标签
        img_name = self.label_df.iloc[idx, 0]+".png"  # 假设第一列是图像文件名
        target_class=self.label_df.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        # 加载图像和标签
        image = Image.open(img_path).convert("RGB")  # 转换为 RGB 图像
        target_segmentation = Image.open(label_path).convert("L")  # 转换为灰度图像，二分类任务中

        # 如果定义了变换操作，则对图像和标签同时进行变换
        if self.transform:
            image = self.transform(image)
            target_segmentation = self.transform(target_segmentation)

        return image, target_segmentation, target_class


class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): 测试集图像文件夹路径
            transform (callable, optional): 图像的变换/标准化操作
        """
        self.img_dir = img_dir
        self.transform = transform if transform is not None else default_transform
        self.img_filenames = sorted(os.listdir(img_dir))  # 获取测试集的所有图像文件名

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 加载图像
        image = Image.open(img_path).convert("RGB")

        # 如果定义了变换操作，则对图像进行变换
        if self.transform:
            image = self.transform(image)

        return image, img_name  # 返回图像和对应的文件名，方便保存预测结果
