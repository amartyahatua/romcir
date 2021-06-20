from sklearn.neighbors import KNeighborsClassifier
import  pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import  pandas as pd

model = SentenceTransformer('distilbert-base-nli-mean-tokens')



filename = 'trained_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

predicted_class = []
f = open("gen_data.txt", encoding="utf-8")
for sentence in f:
    sentence_embeddings = model.encode(sentence)
    predict_class = loaded_model.predict([sentence_embeddings])
    predicted_class.append(predict_class[0])

predicted_class = pd.DataFrame(predicted_class)
predicted_class.to_csv("gendata_predicted_class.csv")
print("Done")