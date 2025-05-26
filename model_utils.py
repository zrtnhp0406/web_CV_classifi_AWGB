import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as tv_models
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torchvision.models as models
# Thiết bị (GPU nếu có, không thì dùng CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sửa: Đường dẫn cố định tới thư mục chứa model
MODEL_DIR = r'C:\Users\ADMIN\Downloads\FruitClassify\FruitClassify\models'

# Tên các lớp
CLASS_NAMES = ['Apple', 'Banana', 'Grape', 'Watermelon']

from efficientnet_pytorch import EfficientNet

def create_resnet():
    """Tạo mô hình ResNet18 cho phân loại"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model

def create_efficientnet():
    """Tạo mô hình EfficientNetB0 cho phân loại"""
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    return model

def load_models():
    """Load 1 model EfficientNet và 1 model ResNet từ thư mục MODEL_DIR"""
    loaded_models = {}

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]

        if len(model_files) == 0:
            logging.warning("Không tìm thấy file .pth nào, sử dụng mô hình dummy.")
            loaded_models['EfficientNet'] = create_dummy_model()
            loaded_models['resnet'] = create_dummy_model()
            return loaded_models

        for model_file in model_files:
            model_path = os.path.join(MODEL_DIR, model_file)
            try:
                if "efficientnet" in model_file.lower():
                    model = create_efficientnet()
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model.eval()
                    loaded_models['EfficientNet'] = model
                    logging.info(f"Đã load EfficientNet từ: {model_file}")
                elif "resnet" in model_file.lower():
                    model = create_resnet()
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model.eval()
                    loaded_models['resnet'] = model
                    logging.info(f"Đã load ResNet từ: {model_file}")
            except Exception as e:
                logging.error(f"Lỗi khi load model {model_file}: {str(e)}")


        return loaded_models

    except Exception as e:
        logging.error(f"Lỗi tổng quát khi load models: {str(e)}")
        return {
            'model1': create_dummy_model(),
            'model2': create_dummy_model()
        }


def create_dummy_model():
    """Mô hình giả cho mục đích test"""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, len(CLASS_NAMES))
        
        def forward(self, x):
            batch_size = x.size(0)
            outputs = torch.randn(batch_size, len(CLASS_NAMES))
            outputs[:, 0] += 1.0  # Ưu tiên Apple
            return outputs
    
    model = DummyModel().to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Tiền xử lý ảnh"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        return image_tensor
    except Exception as e:
        logging.error(f"Lỗi tiền xử lý ảnh: {str(e)}")
        raise

def predict_fruit(models, image_tensor):
    """Chạy dự đoán trên ảnh bằng tất cả các mô hình"""
    predictions = []

    try:
        with torch.no_grad():
            for model_name, model in models.items():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                for i, class_name in enumerate(CLASS_NAMES):
                    confidence = float(probabilities[0][i]) * 100
                    predictions.append({
                        'model': model_name,
                        'class': class_name,
                        'confidence': confidence
                    })

        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions

    except Exception as e:
        logging.error(f"Lỗi dự đoán: {str(e)}")
        raise
