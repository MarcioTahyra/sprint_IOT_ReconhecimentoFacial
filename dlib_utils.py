import dlib
import numpy as np
import os
import pickle
from base64 import b64decode
from io import BytesIO
from PIL import Image

PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
USERS_FILE = "users.pkl"
THRESH = 0.5

try:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTOR)
    rec = dlib.face_recognition_model_v1(RECOG)
except RuntimeError as e:
    print(f"ERRO: Não foi possível carregar os modelos Dlib: {e}")
    print(
        "Certifique-se de que 'shape_predictor_5_face_landmarks.dat' e 'dlib_face_recognition_resnet_model_v1.dat' estão no diretório.")
    exit()

db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}
users = pickle.load(open(USERS_FILE, "rb")) if os.path.exists(USERS_FILE) else {}



def get_embedding(img, rect):
    shape = sp(img, rect)
    chip = dlib.get_face_chip(img, shape)
    chip_np = np.array(chip)
    return np.array(rec.compute_face_descriptor(chip_np), dtype=np.float32)


def reconhecer(nome, vec):
    if nome not in db:
        return False
    dist = np.linalg.norm(vec - db[nome])
    return dist <= THRESH


def salvar_usuario(nome, senha, vec):
    users[nome] = senha
    db[nome] = vec
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)
    return True


def processar_frame_para_embedding(base64_img):

    try:
        if ";base64," in base64_img:
            _, encoded = base64_img.split(",", 1)
        else:
            encoded = base64_img

        img_bytes = b64decode(encoded)
        img = Image.open(BytesIO(img_bytes))
        img_np = np.array(img.convert("RGB"))



        rects = detector(img_np, 1)

        if rects:
            rect = rects[0]
            vec = get_embedding(img_np, rect)
            return vec
        else:
            return None

    except Exception as e:
        print(f"Erro ao processar frame para embedding: {e}")
        return None