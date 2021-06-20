from sklearn.neighbors import KNeighborsClassifier
import  pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


model = SentenceTransformer('distilbert-base-nli-mean-tokens')

df = pd.read_csv('claim_encode.csv')


df = df.values.tolist()

features = []
classes = []
evidences = []
for i in range(len(df)):
    features.append(df[i][0:768])

for i in range(len(df)):
    classes.append(df[i][768])

print("Original data loading done")
Data_2 = pd.read_csv('gen_data.csv')
Data_2 = Data_2.values.tolist()

gen_clss = pd.read_csv('gendata_predicted_class.csv', index_col=None)
gen_clss = gen_clss.values.tolist()

for i in range(len(gen_clss)):
    sentence = Data_2[i]
    sentence_embeddings = model.encode(sentence)
    features.append(sentence_embeddings[0])
    classes.append(gen_clss[i][1])


print("Generated data loading done")


model = KNeighborsClassifier(n_neighbors=10)
model.fit(features, classes)

print("Training done")

filename = 'trained_model_train_gen_data.sav'
pickle.dump(model, open(filename, 'wb'))