import argparse
import os
import zipfile

import pandas as pd
import torch
from PIL import Image

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainImageDataset, TestImageDataset
from loss import combined_loss
from model import UNetWithClassHead


def main(args):
    print(args)
    device = torch.device(args.device)

    # Load datasets
    train_dir=os.path.join(args.dataset_dir, 'train')
    train_dataset = TrainImageDataset(
        img_dir=os.path.join(train_dir, 'img'),
        label_dir=os.path.join(train_dir, 'label'),
        label_csv=os.path.join(train_dir, 'label.csv'),
    )
    test_dataset = TestImageDataset(img_dir=os.path.join(os.path.join(args.dataset_dir, 'test'),'img'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNetWithClassHead(n_channels=3, n_classes=args.num_classes,
                              n_segmentation_classes=args.num_segmentation_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    model.to(device)
    total_loss = 0

    for images, target_segmentation, target_class in tqdm(train_loader):
        images = images.to(device)
        target_segmentation = target_segmentation.to(device)
        target_class = target_class.to(device)

        # 前向传播
        seg_output, class_output = model(images)

        # 计算损失
        loss, seg_loss, class_loss = combined_loss(seg_output, class_output, target_segmentation,
                                                   target_class)  # target_class 如果有则传入
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"训练平均损失: {avg_loss:.4f}")

    # TEST
    model.eval()
    predictions = []

    label_dir = os.path.join(args.output_dir, 'label')
    os.makedirs(label_dir, exist_ok=True)  # 创建保存预测结果的目录

    with torch.no_grad():
        for images, img_names in tqdm(test_loader):
            images = images.to(device)

            # 预测分割结果
            seg_output, class_output = model(images)

            # 处理分割结果
            predicted_mask = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy()

            # 保存预测分割掩码
            img_filename = img_names[0]
            save_path = os.path.join(label_dir, img_filename)
            save_mask(predicted_mask, save_path)

            # 将文件名和标签信息加入到 CSV 列表中
            predictions.append({'case': img_filename, 'prob': class_output.squeeze(0).cpu().numpy()[0]})

    # 保存 label.csv
    save_csv(predictions, args.output_dir, "label.csv")

    # 将结果压缩为 submit.zip
    zip_output(args.output_dir, 'submit.zip')


# 保存分割掩码为 PNG 图像
def save_mask(mask, path):
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    mask_img = Image.fromarray(mask.astype('uint8'))
    mask_img.save(path)


# 保存 label.csv
def save_csv(predictions, output_dir: str, csv_filename):
    df = pd.DataFrame(predictions)
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)


# 将目录打包为 zip 文件
def zip_output(output_dir, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目录下的所有文件，进行压缩
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = str(os.path.join(root, file))
                zipf.write(file_path, os.path.relpath(file_path, output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_dir', type=str, default="../tcdata")
    parser.add_argument('--output_dir', type=str, default="./submit")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_segmentation_classes', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    opt = parser.parse_args()

    main(opt)
