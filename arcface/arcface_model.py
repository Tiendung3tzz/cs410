import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from arcface.mtcnn import *
from PIL import Image, ImageDraw, ImageFont

NUM_CLASSES = 4         # Số lớp (tương ứng với số thư mục)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Name = ["Trấn Thành","Lê Giang", "Tuấn Trần", "Uyển Ân"]
font = ImageFont.truetype("arial.ttf", size=25)

def normalize_to_range(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def load_model():
    model = models.resnet50(pretrained=True)  # Vẫn dùng ResNet-50
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Đảm bảo fully connected layer khớp với số lớp
    model.load_state_dict(torch.load("/kaggle/input/bogia-embedding/arcface_finetuned.pth"))  # Tải lại các tham số đã huấn luyện
    model = model.to(DEVICE)  # Chuyển mô hình về device (GPU hoặc CPU)
    model.eval()  
    return model

st.cache_resource
def get_arcface():
    return load_model(),get_mtcnn(DEVICE)

def get_name_img(ent_results):
    text_name = set()
    for result in ent_results:
        text_name.add(result['word'])
    return text_name

def max_distance(normalized_array):
    max_index = np.argmax(normalized_array)
    return max_index

def arcface_run(retrieved_image_files,mtcnn,model,distances,ent_results,thresod):
    normalized_array = normalize_to_range(distances)
    prename = set()
    text_name = get_name_img(ent_results)
    j = 0
    conf = 0
    for image_test in retrieved_image_files:
    
    # Transform cho ArcFace
        arcface_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # Input and output paths
        input_image_path = image_test
        output_image_path = f"/kaggle/working/cs410/output/output_{j}.jpg"
        
        # Load image
        image = Image.open(input_image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)
        
        # Perform face detection with MTCNN
        boxes, probs = mtcnn.detect(image)
        
        # Filter out faces with low probability
        if boxes is not None:
            boxes, probs = boxes[probs > thresod], probs[probs > thresod]  # Lọc các khuôn mặt với độ tin cậy > 0.9
        
        # Annotate and predict labels
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        if boxes is not None:
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box)
                
                # Crop the face region
                face_crop = image.crop((x_min, y_min, x_max, y_max))
                
                # Transform the face crop for ArcFace
                face_tensor = arcface_transform(face_crop).unsqueeze(0).to(DEVICE)
                
                # Predict label with ArcFace
                with torch.no_grad():
                    embedding = model(face_tensor)
                    predicted_label = torch.argmax(embedding).item()
                    confidence_score = torch.softmax(embedding, dim=1).max().item()
                    if Name[predicted_label] in text_name:
                        if (Name[predicted_label] not in prename):
                            prename.add(Name[predicted_label])
                            print(image_test)
                            normalized_array[j]=normalized_array[j] + confidence_score;
                            print(normalized_array[j],':',conf,':',predicted_label)
                            conf = confidence_score
                        elif (conf < confidence_score):
                            print(image_test)
                            normalized_array[j]=normalized_array[j] + confidence_score - conf;
                            print(normalized_array[j],'conf:',conf,'confidence_score:',confidence_score,'label:',predicted_label)
                            conf = confidence_score
                # Draw bounding box and label on the image
                draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=3)
                draw.text((x_min, y_min - 15), f"ID: {predicted_label} {confidence_score} ", fill="red")
            
            # Save and show the annotated image
            annotated_image.save(output_image_path)
        conf = 0
        j+=1
    img_final = f"/kaggle/working/cs410/output/output_{max_distance(normalized_array)}.jpg"
    return max_distance(normalized_array), img_final
