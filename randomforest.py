import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import  pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


model = SentenceTransformer('paraphrase-distilroberta-base-v1')
df = pd.read_csv('data/Data.csv',  nrows=None)

df_test = pd.read_csv('data/Test_set_paper_for_testing.csv',  nrows=None)

df_claims = df['claims']
df_class = df['classes']

df_claims_test = df_test['claims']
df_class_test = df_test['classes']

encoded = []
encoded_test = []

for corp in df_claims:
    encode = model.encode(corp)
    encoded.append(encode)

for corp in df_claims_test:
    encode = model.encode(corp)
    encoded_test.append(encode)


X = encoded
y= df_class

clf = RandomForestClassifier()
clf.fit(X, y)


rf_result = clf.predict(encoded_test)


precision_metric_micro = precision_score(df_class_test, rf_result, average = "micro")
recall_metric_micro = recall_score(df_class_test, rf_result, average = "micro")
accuracy_metric_micro = accuracy_score(df_class_test, rf_result)
f1_metric_micro = f1_score(df_class_test, rf_result, average = "micro")

print("precision_metric micro", precision_metric_micro)
print("recall_metric micro", recall_metric_micro)
print("accuracy_metric micro", accuracy_metric_micro)
print("f1_metric micro", f1_metric_micro)