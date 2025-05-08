import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import VGG
#from torchviz import make_dot
from torchsummary import summary  # 모델 아키텍처 요약을 위한 라이브러리
import matplotlib.pyplot as plt
import os

# 장치 설정: GPU 사용 가능 시 GPU로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 경로 설정
train_data_dir = r'C:\Users\computer\Desktop\Code\KISTI\Flood\dataset\train'
test_data_dir = r'C:\Users\computer\Desktop\Code\KISTI\Flood\dataset\test'
validation_data_dir = r'C:\Users\computer\Desktop\Code\KISTI\Flood\dataset\validation'

# 이미지 변환: 센터 크롭 후 225*225 리사이즈
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.center_crop(img, min(img.size))),
    transforms.Resize((225, 225)),
    transforms.ToTensor(),
    # 색에 대해 정규화 추가
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 파일 유효성 검사 함수 정의 (유효한 확장자 체크)
def is_valid_image_file(file_path):
    return file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))

# 훈련, 테스트, 검증 데이터셋 설정, 유효한 이미지 파일만 포함
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform, is_valid_file=is_valid_image_file)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform, is_valid_file=is_valid_image_file)
val_dataset = datasets.ImageFolder(root=validation_data_dir, transform=transform, is_valid_file=is_valid_image_file)

# 데이터로더 설정: 배치 크기 32, 훈련 데이터는 셔플
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#사용자 정의 VGG16 모델 (사전 훈련 가중치 없음)
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

# VGG 레이어 생성 함수: 배치 정규화 옵션 포함
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 사용자 정의 VGG16 모델 클래스
class VGG16Custom(VGG):
    def __init__(self, num_classes=5):
        super(VGG16Custom, self).__init__(make_layers(cfg['D']), num_classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes)
        )

# 모델, 옵티마이저, 손실 함수 정의
model = VGG16Custom(num_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 체크포인트 저장 함수
def save_checkpoint(state, filename=r'C:\Users\computer\Desktop\Code\KISTI\Flood\checkpoint\vgg16_best_checkpoint_v2.pth'):
    torch.save(state, filename)

# 체크포인트 로드 함수
def load_checkpoint(filename=r'C:\Users\computer\Desktop\Code\KISTI\Flood\checkpoint\vgg16_best_checkpoint_v2.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    return epoch, accuracy

# 모델 평가 함수 (정확도 반환)
def evaluate_model(loader):
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 정확도 계산
        accuracy = 100 * correct / total
        return accuracy

# 모델 훈련 함수 (Train, Validation Accuracy 포함)
def train_model(num_epochs, best_accuracy):
    model.train()  # 모델을 훈련 모드로 설정
    train_accuracies, val_accuracies, train_losses = [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 기울기 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 훈련 정확도 계산
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = evaluate_model(val_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss}, Train Accuracy: {train_accuracy}%, Validation Accuracy: {val_accuracy}%')

        # 새로운 최고 성능일 때 체크포인트 저장
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            })

    return train_losses, train_accuracies, val_accuracies

# 학습 그래프 저장 함수
def plot_training_graph(train_losses, val_accuracies, save_path):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.savefig(os.path.join(save_path, "training_graph.png"))
    plt.close()

def visualize_model_architecture():
    summary(model, (3, 125, 125))  # 모델 아키텍처 요약 출력

# 테스트 정확도 출력 함수
def test_model():
    test_accuracy = evaluate_model(test_loader)
    print(f"Test Accuracy: {test_accuracy}%")
    return test_accuracy

# 모델 훈련 및 평가 실행
if __name__ == "__main__":
    save_path = r"C:\Users\computer\Desktop\hi"  # 이미지 저장 경로 설정
    num_epochs = 20
    best_accuracy = 0.0

    # 모델 훈련
    train_losses, train_accuracies, val_accuracies = train_model(num_epochs, best_accuracy)
    
    # 학습 그래프 저장
    plot_training_graph(train_losses, val_accuracies, save_path)

    # 모델 아키텍처 시각화
    visualize_model_architecture()

    # 테스트 정확도 출력
    test_accuracy = test_model()

    # Train/Validation/Test 데이터셋 크기 출력
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")