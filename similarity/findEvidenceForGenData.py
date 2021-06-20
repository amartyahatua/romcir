from sklearn.neighbors import KNeighborsClassifier
import  pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import  pandas as pd


## This program helps to find the evidence of generated data

model = SentenceTransformer('distilbert-base-nli-mean-tokens')


## Loading claim_evidence 
df = pd.read_csv('claim_encode.csv')
df = df.values.tolist()

features = []
classes = []
evidences = []
for i in range(len(df)):
    features.append(df[i][0:768])

for i in range(len(df)):
    classes.append(df[i][768])

for i in range(len(df)):
    evidences.append(df[i][769])

## Loading the generated data
f = open("gen_data.txt", encoding="utf-8")
evidence_gen = []

print("Processing generated data")
for sentence in f:
    max_index = -1
    max_dis = -1
    sentence_embeddings = model.encode(sentence)
    aa = sentence_embeddings.reshape(1, 768)
    for i in range(len(features)):
        arr = np.array(features[i])
        ba = arr.reshape(1, -1)
        cos_lib = cosine_similarity(aa, ba)
        if(cos_lib > max_dis):
            max_dis = cos_lib
            max_index = i
    evidence_gen.append(evidences[max_index])

print("Done Processing generated data")
evidence_gen = pd.DataFrame(evidence_gen)
evidence_gen.to_csv("gendata_data_evidence.csv")
print("Done")