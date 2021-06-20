from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import  pandas as pd

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
df = pd.read_csv('claim_class.csv')
encodeList = []
claims = df['claim']
evidences = df['evidence']


for i in range(len(claims)):
    claim = claims[i]
    claim_encode = model.encode(claim)
    encodeList.append(claim_encode)

encodeList = pd.DataFrame(encodeList)
df = pd.concat([encodeList, df['0'], evidences], axis= 1)
df.to_csv('claim_encode.csv', index=None)
