import json
import re
import shutil
from difflib import SequenceMatcher

import psycopg2

from tqdm import tqdm
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request

def load_train_valid_split(Config):

        X = []
        y = []

        for file in tqdm(os.listdir(Config['X_path'])):

            img=cv2.imread(Config['X_path']+file)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(256,256))
            X.append(img)
            
            img=cv2.imread(Config['y_path']+file)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img,(256,256))
            y.append(list(img))

        X=np.array(X)
        y=np.array(y,dtype=np.bool)

        X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=Config['valid_size'])

        # Put img used for training in their respective folders
        for i, img in enumerate(X_train):
            cv2.imwrite(Config['X_train_path']+"{}.jpg".format(i), img)

        for i, img in enumerate(X_valid):
            cv2.imwrite(Config['X_valid_path']+"{}.jpg".format(i), img)

        return X_train,X_valid,y_train,y_valid

def clean_training_folders(Config):

    for file in os.listdir(Config['X_train_path']):
        os.remove(Config['X_train_path']+file)
    for file in os.listdir(Config['X_valid_path']):
        os.remove(Config['X_valid_path']+file)

def clean_results_folder(Config):

    # rm results folder and all folder / files in it
    shutil.rmtree(r"{}".format(Config['results_path']))
    # make a new one
    os.mkdir(Config['results_path'])

def load_label_to_read(Config):

        fileNames = []
        srcs = []
        X = []

        for file in tqdm(os.listdir(Config['to_read_path'])):

            #buid a file structure for results
            filename = file.rsplit( ".", 1 )[0]
            fileNames.append(filename)
            parent_dir = Config['results_path']
            path = os.path.join(parent_dir, filename)
            os.mkdir(path)

            src=cv2.imread(Config['to_read_path']+file)
            srcs.append(src)
            img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(256,256))
            X.append(img)

            cv2.imwrite(path + "/" + "0_src.jpg", src)
            cv2.imwrite(path + "/" + "1_unet.jpg", img)

        X=np.array(X)

        return X, srcs, fileNames

def img_url_to_input_unet(url):
    
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    X=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X=cv2.resize(X,(256,256))

    return np.array([X]), img


def normalize_text(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def split_and_normalize(string, delimiter=";"):
    return [normalize_text(token) for token in string.split(delimiter)]


def match_score(search_tokens, wine):
    fields = f"{wine[9]} {wine[2]} {wine[10]} {wine[7]} {wine[5]} {wine[6]}"
    normalized_fields = normalize_text(fields)

    match_count = sum(1 for token in search_tokens if token in normalized_fields)

    similarity = SequenceMatcher(None, " ".join(search_tokens), normalized_fields).ratio()

    return match_count, similarity


def find_wine_by_etiquette(search_string):
    search_tokens = split_and_normalize(search_string)
    results = []

    wine_db = fetch_all_wines(establish_db_connection())

    for wine in wine_db:
        match_count, similarity = match_score(search_tokens, wine)
        results.append((match_count, similarity, wine))

    results.sort(key=lambda x: (-x[0], -x[1]))
    return results[0][2] if results else None

def establish_db_connection():
    conn = psycopg2.connect(
        dbname="fine-wine-db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    return conn

def fetch_all_wines(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM wines")
    rows = cur.fetchall()
    cur.close()
    return rows
