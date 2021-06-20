import  pandas as pd
#claim,evidence,class
df = pd.read_csv('Data.csv')
df_claim = df['claim']
df_evidence = df['evidence']

df_claim = df_claim.values.tolist()
df_evidence = df_evidence.values.tolist()

data = []

for i in range(len(df_claim)):
    data.append(df_claim[i])
    data.append(df_evidence[i])

data = pd.DataFrame(data)
data.to_csv('Data_LeakGAN.text', index=False)
