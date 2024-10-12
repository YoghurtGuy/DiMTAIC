import argparse

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from app.dataset import TrainImageDataset, TestImageDataset
from app.loss import combined_loss
from app.model import UNetWithClassHead


def main(args):
    print(args)
    device = torch.device(args.device)

    # Load datasets
    train_dataset = TrainImageDataset(
        img_dir=args.train_img_dir,
        label_dir=args.train_label_dir,
        label_csv=args.train_label_csv,
    )
    test_dataset = TestImageDataset(img_dir=args.test_img_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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
        loss, seg_loss, class_loss = combined_loss(seg_output, class_output, target_segmentation, target_class)  # target_class 如果有则传入
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"训练平均损失: {avg_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--train_img_dir', type=str, default="tcdata/train/img")
    parser.add_argument('--train_label_dir', type=str, default="tcdata/train/label")
    parser.add_argument('--train_label_csv', type=str, default="tcdata/train/label.csv")
    parser.add_argument('--test_img_dir', type=str, default="tcdata/test/img")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_segmentation_classes', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    opt = parser.parse_args()

    main(opt)
