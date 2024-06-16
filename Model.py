from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
import faiss
import torch
import timm
import numpy as np
import psycopg2
from PIL import Image
import logging
from io import BytesIO

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI()

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 모델 로드 (EfficientNet B3)
model = timm.create_model('efficientnet_b3', pretrained=True)
model.to(device)
model.eval()

def load_image(image_data):
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    return image

def get_image_embedding(image_tensor):
    with torch.no_grad():
        embedding = model.forward_features(image_tensor)
    return embedding.cpu().numpy()

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# FAISS 인덱스 파일 로드
index = faiss.read_index("faiss_index2.index")
logger.info(f"FAISS index loaded with dimension: {index.d}")

# Cloud SQL 연결 설정
db_user = "postgres"
db_pass = "#DG[b-#3o/oxH8&"
db_name = "stylefinder"
db_host = "34.64.97.50"  # 공개 IP 주소

# 데이터베이스 연결 함수
def get_db_connection():
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_pass,
        host=db_host
    )
    return conn

# FastAPI 엔드포인트 설정
@app.post("/find/")
async def find_image(multipart_file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 읽고 이미지로 변환
        logger.info("Reading uploaded file")
        contents = await multipart_file.read()
        image = load_image(contents)

        logger.info("Image decoded and preprocessed successfully")

        # 이미지를 벡터로 변환
        query_vector = get_image_embedding(image)
        query_vector = query_vector.reshape(1, -1)
        query_vector = normalize_vectors(query_vector)

        logger.info(f"Query vector shape: {query_vector.shape}")

        if query_vector.shape[1] != index.d:
            logger.error(f"Query vector size {query_vector.shape[1]} does not match FAISS index dimension {index.d}")
            raise HTTPException(status_code=400, detail="Query vector size does not match FAISS index dimension")

        # FAISS 인덱스에서 검색 수행
        k = 5  # 반환할 이웃의 수
        D, I = index.search(query_vector, k)  # 거리 D, 인덱스 I
        logger.info(f"Search results: {I}")

        # SQL 쿼리 수행
        result_list = []
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            for idx in I[0]:
                idx_int = int(idx)  # numpy.int64를 int로 변환
                logger.info(f"Searching for ID {idx_int} in the database")
                cursor.execute("SELECT * FROM clothes WHERE id=%s", (idx_int,))
                row = cursor.fetchone()
                if row:
                    result_list.append({
                        "id": row[0],
                        "data": row[1:]  # 실제 반환할 데이터 열로 변경하세요
                    })
                else:
                    logger.info(f"No result found for ID {idx_int}")
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Database query failed")

        # JSON 응답 반환
        logger.info(f"Returning {len(result_list)} results")
        return {"results": result_list}
    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
