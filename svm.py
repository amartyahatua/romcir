
from sentence_transformers import SentenceTransformer
import  pandas as pd
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle


sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
df = pd.read_csv('data/Data.csv',  nrows=None)

df_claim = df['claims']
#df_evidence = df['evidence']
df_class = df['classes']

df_claim = df_claim.values.tolist()
#df_evidence = df_evidence.values.tolist()
df_class = df_class.values.tolist()


df_claim_encode = []
df_evidence_encode = []

testData = pd.read_csv('data/gen_data_LaTexGAN.csv')
X_test = testData['claims']
y_test = testData['classes']

X_test = X_test.values.tolist()
y_test = y_test.values.tolist()

for claim in df_claim:
    temp = sbert_model.encode(claim)
    df_claim_encode.append(temp)

X_test_encode = []
for test_data in X_test:
    temp = sbert_model.encode(test_data)
    X_test_encode.append(temp)


claim_evidence = []
for i in range(len(df_claim)):
    clm = df_claim[i]
    #eve = df_evidence[i]
    temp = clm
    claim_evidence.append(temp)
df_claim_evidence_encode = []
for evidence in claim_evidence:
    temp = sbert_model.encode(evidence)
    df_claim_evidence_encode.append(temp)


df_claim_evidence_encode = pd.DataFrame(df_claim_evidence_encode)

clf = svm.SVC()
print("Model fitting started ....")
model = clf.set_params(kernel='rbf').fit(df_claim_evidence_encode, df_class)
svc_predictions = model.predict(X_test_encode)

precision_metric = precision_score(y_test, svc_predictions, average = "macro")
recall_metric = recall_score(y_test, svc_predictions, average = "macro")
accuracy_metric = accuracy_score(y_test, svc_predictions)
f1_metric = f1_score(y_test, svc_predictions, average = "macro")


svc_predictions = pd.DataFrame(svc_predictions)
svc_predictions.to_csv("Precition_Leak_GAN_Result.csv")

filename = 'svm_model.sav'
pickle.dump(model, open(filename, 'wb'))



print("precision_metric", precision_metric)
print("recall_metric", recall_metric)
print("accuracy_metric", accuracy_metric)
print("f1_metric", f1_metric)



precision_metric_micro = precision_score(y_test, svc_predictions, average = "micro")
recall_metric_micro = recall_score(y_test, svc_predictions, average = "micro")
accuracy_metric_micro = accuracy_score(y_test, svc_predictions)
f1_metric_micro = f1_score(y_test, svc_predictions, average = "micro")

svc_predictions = pd.DataFrame(svc_predictions)
svc_predictions.to_csv("LaTextGAN_Prediction.csv")


print("precision_metric micro", precision_metric_micro)
print("recall_metric micro", recall_metric_micro)
print("accuracy_metric micro", accuracy_metric_micro)
print("f1_metric micro", f1_metric_micro)


