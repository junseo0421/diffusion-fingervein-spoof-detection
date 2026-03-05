import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.ConvNeXt import convnext_small
from os.path import join

from torch.utils.tensorboard import SummaryWriter  # ★ TensorBoard
from tqdm import tqdm  # ★ 진행바

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 1. Dataset 정의
# ==========================
class RealFakeDataset(Dataset):
    """
    real 이미지와 fake 이미지를 한 번에 관리하는 Dataset.
    label: real=0, fake=1
    """
    def __init__(self, real_paths, fake_paths, transform=None):
        self.img_paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]

        # Finger-vein 이미지는 대부분 grayscale이지만,
        # ConvNeXt는 3채널을 기대하므로 RGB로 맞춰줌.
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label


# ==========================
# 2. 경로 모으기 & train/valid/test split (6:2:2)
# ==========================
def build_datasets(
    real_root="Datasets/images/t1_kj",
    fake_root="./logs/HKPU_1/generated_images",
    train_ratio=0.6,   # 60%
    val_ratio=0.2,     # 20% (나머지 20%는 test)
    img_size=64,  # 224 > 64
):
    # real 이미지 경로
    real_paths = glob(os.path.join(real_root, "*", "*"))
    real_paths = [p for p in real_paths if p.lower().endswith((".png", ".jpg", ".bmp"))]

    # fake 이미지 경로
    fake_paths = glob(os.path.join(fake_root, "*.png"))
    fake_paths = [p for p in fake_paths if p.lower().endswith((".png", ".jpg", ".bmp"))]

    print(f"# real images: {len(real_paths)}")
    print(f"# fake images: {len(fake_paths)}")

    n = min(len(real_paths), len(fake_paths))
    if n == 0:
        raise ValueError("real 또는 fake 이미지가 하나도 없습니다. 경로를 다시 확인하세요.")

    # real / fake 개수 맞춰서 잘라줌
    real_paths = real_paths[:n]
    fake_paths = fake_paths[:n]

    # random.shuffle(real_paths)
    # random.shuffle(fake_paths)

    # 비율 체크 (train + val <= 1.0 이어야 함)
    assert 0.0 < train_ratio < 1.0
    assert 0.0 <= val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0 or abs(train_ratio + val_ratio - 1.0) < 1e-6

    # 인덱스 계산: 6:2:2
    r_train = int(len(real_paths) * train_ratio)
    r_val   = int(len(real_paths) * (train_ratio + val_ratio))

    f_train = int(len(fake_paths) * train_ratio)
    f_val   = int(len(fake_paths) * (train_ratio + val_ratio))

    train_real = real_paths[:r_train]
    val_real   = real_paths[r_train:r_val]
    test_real  = real_paths[r_val:]

    train_fake = fake_paths[:f_train]
    val_fake   = fake_paths[f_train:f_val]
    test_fake  = fake_paths[f_val:]

    print(f"train real: {len(train_real)}, val real: {len(val_real)}, test real: {len(test_real)}")
    print(f"train fake: {len(train_fake)}, val fake: {len(val_fake)}, test fake: {len(test_fake)}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_dataset = RealFakeDataset(train_real, train_fake, transform=train_transform)
    val_dataset = RealFakeDataset(val_real, val_fake, transform=eval_transform)
    test_dataset = RealFakeDataset(test_real, test_fake, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


# ==========================
# 3. Train / Eval 루프
# ==========================
def train_one_epoch(model, criterion, optimizer, dataloader, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 진행바
    progress_bar = tqdm(
        dataloader,
        desc=f"Train [{epoch}/{total_epochs}]",
        ncols=100
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)            # (B, 2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += batch_size

        # 진행바에 현재까지의 평균 loss / acc 표시
        avg_loss = running_loss / total
        acc = correct / total if total > 0 else 0.0
        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{acc*100:.2f}%"
        )

    avg_loss = running_loss / total
    acc = correct / total
    print(f"[Train] Epoch [{epoch}/{total_epochs}] Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%")
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, criterion, dataloader, epoch=None, total_epochs=None, mode="Valid"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    if epoch is not None:
        print(f"[{mode}] Epoch [{epoch}/{total_epochs}] Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%")
    else:
        print(f"[{mode}] Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%")

    return avg_loss, acc


# ==========================
# 4. 메인: ConvNeXt-small 학습 + intra-DB test(6:2:2)
# ==========================
def main():
    # ----- 하이퍼파라미터 -----
    real_root = "./datasets/images/t1_kj"
    fake_root = r"C:\Users\8138\PycharmProjects\pytorch-ddpm-master\logs\HKPU_1\generated_images_last"
    img_size = 64  # 224 > 64
    batch_size = 16
    num_epochs = 400
    train_ratio = 0.6
    val_ratio = 0.2
    lr = 1e-4


    # ----- TensorBoard Writer -----
    log_dir = "./runs/convnext_real_fake_64_head_scratch"  # 원하는 경로로 변경 가능
    save_path = join(log_dir, "convnext_real_fake_best.pth")
    writer = SummaryWriter(log_dir=log_dir)

    # ----- Dataset / DataLoader -----
    train_dataset, val_dataset, test_dataset = build_datasets(
        real_root=real_root,
        fake_root=fake_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ----- 모델 준비 -----
    model = convnext_small(
        pretrained=True,
        in_22k=False,
        num_classes=2,    # real / fake
        in_chans=3,
    )
    model = model.to(device)

    # 1) 일단 전부 freeze
    for p in model.parameters():
        p.requires_grad = False

    # 2) head 부분만 학습 가능하게 풀어주기
    for name, p in model.named_parameters():
        if name.startswith("head"):  # convnext 구현상 head가 마지막 classifier
            p.requires_grad = True

    # 3) optimizer 는 requires_grad=True 인 파라미터만
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5,  # 조금 낮게 시작 (1e-4가 너무 빠르면)
        weight_decay=5e-4,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # loss가 줄어들지 않을 때
        factor=0.3,  # lr *= 0.3
        patience=3,  # 3 epoch 연속 개선 없으면 줄이기
        verbose=True
    )

    best_val_acc = 0.0
    best_val_loss = 1e3

    # ----- 학습 루프 -----
    # for epoch in range(1, num_epochs + 1):
    #     # Train
    #     train_loss, train_acc = train_one_epoch(
    #         model, criterion, optimizer, train_loader, epoch, num_epochs
    #     )
    #     # TensorBoard 기록 (Train)
    #     writer.add_scalar("Loss/train", train_loss, epoch)
    #     writer.add_scalar("Acc/train",  train_acc,  epoch)
    #
    #     # Validation
    #     val_loss, val_acc = evaluate(
    #         model, criterion, val_loader, epoch, num_epochs, mode="Valid"
    #     )
    #     # TensorBoard 기록 (Valid)
    #     writer.add_scalar("Loss/val", val_loss, epoch)
    #     writer.add_scalar("Acc/val",  val_acc,  epoch)
    #
    #     # lr 스케줄러
    #     scheduler.step(val_loss)
    #
    #     # best model 저장
    #     if val_acc >= best_val_acc and val_loss <= best_val_loss:
    #         best_val_acc = val_acc
    #         best_val_loss = val_loss
    #         torch.save({
    #             "model_state": model.state_dict(),
    #             "val_acc": best_val_acc,
    #             "val_loss": best_val_loss,
    #             "epoch": epoch,
    #         }, save_path)
    #         print(f"✔ Best model updated: acc={best_val_acc*100:.2f}%, saved to {save_path}")
    #
    # print("Training finished.")
    # print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    # ----- best 모델 로드 후 test 평가 -----
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("=== Final Test Evaluation (6:2:2 split) ===")
        test_loss, test_acc = evaluate(model, criterion, test_loader, mode="Test")

        # TensorBoard 기록 (Test) — step은 num_epochs+1로 줌
        writer.add_scalar("Loss/test", test_loss, num_epochs + 1)
        writer.add_scalar("Acc/test",  test_acc,  num_epochs + 1)
    else:
        print("Warning: best ckpt not found. Test evaluation skipped.")

    writer.close()



# ==========================
# 5. 별도 cross-DB test용 함수
# ==========================
def test_model(test_root_real, test_root_fake, model_ckpt, img_size=64, batch_size=16):
    """
    별도의 test real/fake 폴더가 있을 때 사용하는 함수.
    - test_root_real: test용 real finger-vein 이미지 폴더
    - test_root_fake: test용 fake 이미지 폴더 (DDPM 등으로 생성)
    """
    # 경로 수집
    real_paths = glob(os.path.join(test_root_real, "*", "*"))
    real_paths = [p for p in real_paths if p.lower().endswith((".png", ".jpg", ".bmp"))]

    fake_paths = glob(os.path.join(test_root_fake, "*.png"))
    fake_paths = [p for p in fake_paths if p.lower().endswith((".png", ".jpg", ".bmp"))]

    n = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:n]
    fake_paths = fake_paths[:n]

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_dataset = RealFakeDataset(real_paths, fake_paths, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 모델 로드
    model = convnext_small(
        pretrained=True,
        in_22k=False,
        num_classes=2,
        in_chans=3,
    ).to(device)

    ckpt = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.CrossEntropyLoss()
    evaluate(model, criterion, test_loader, mode="Test")


if __name__ == "__main__":
    main()

    # cross-DB 테스트 하고 싶으면 별도로:
    # test_model("Datasets/images/t1_kj_test",
    #            "./logs/HKPU_1/generated_images_test",
    #            "convnext_real_fake_best.pth")
