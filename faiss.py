import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import faiss
import torch
import os
import timm
import re
import pickle

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 파일 경로 가져오기 함수
def get_all_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 폴더 경로 설정
folder_path = './images'  # 다운로드한 이미지가 저장된 경로


# 모든 이미지 파일 경로 가져오기
image_paths = get_all_image_paths(folder_path)

# 이미지 경로 데이터프레임 생성
df = pd.DataFrame(image_paths, columns=['image_path'])

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 로드 및 전처리 함수
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    return image

# 모델 로드 (EfficientNet B3)
model = timm.create_model('efficientnet_b3', pretrained=True)
model.to(device)
model.eval()

def get_image_embedding(image_tensor):
    with torch.no_grad():
        embedding = model.forward_features(image_tensor)  # 모델에 따라 .forward_features()를 사용해야 할 수 있음
    return embedding.cpu().numpy()

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# 이미지 임베딩 계산 및 정규화
embeddings = []
image_ids = []

for path in df['image_path']:
    image = load_image(path)
    embedding = get_image_embedding(image)

    # 이미지 파일 이름에서 ID 추출 (숫자 부분만)
    image_id = int(re.findall(r'\d+', os.path.basename(path))[0])

    embeddings.append(embedding)
    image_ids.append(image_id)

# 4차원 임베딩 벡터를 2차원으로 변환
embeddings = np.vstack(embeddings)
n, c, h, w = embeddings.shape
embeddings = embeddings.reshape(n, c * h * w)

# 정규화
embeddings = normalize_vectors(embeddings)

# 형상 출력
print(f"Embeddings shape: {embeddings.shape}")

vector_dimension = embeddings.shape[1]

# FAISS 인덱스 생성 및 추가
try:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vector_dimension))
    index.add_with_ids(embeddings, np.array(image_ids))
    print("FAISS index has been created and saved successfully in memory.")
except ValueError as e:
    print(f"Error: {e}")
    print("Ensure that the embeddings array is 2D and has the shape (n, d).")

# 인덱스 저장
faiss.write_index(index, 'faiss_index2.index')
print("FAISS index has been saved to 'faiss_index.bin'.")

# 인덱스와 이미지 ID 매핑 저장
with open('image_ids.pkl', 'wb') as f:
    pickle.dump(image_ids, f)
print("Image IDs have been saved to 'image_ids.pkl'.")

print(image_ids)
print(faiss.vector_to_array(index.id_map))

# 인덱스 로드 (선택 사항)
# index = faiss.read_index('faiss_index.bin')
# with open('image_ids.pkl', 'rb') as f:
#     image_ids = pickle.load(f)
# print("FAISS index and image IDs have been loaded.")

search_image_path = './images/training/381.jpg'
search_list = []
search_image = load_image(search_image_path)
search_vector = get_image_embedding(search_image)
search_list.append(search_vector)

search_list = np.vstack(search_list)
n, c, h, w = search_list.shape
search_list = search_list.reshape(n, c * h * w)

search_list = normalize_vectors(search_list)


search_dimension = search_list.shape[1]

D, I = index.search(search_list, k=3)

# 결과 출력
top_indices = I[0]
top_results = D[0]

print(top_indices)
print(top_results)

D, I = index.search(embeddings[0],k=3)

print(D[0],I[0])