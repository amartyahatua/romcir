import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sentence_transformers import SentenceTransformer
import  pickle
import seaborn as sns
from matplotlib import pyplot as plt
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale


RS = 25111993

# Importing matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib


# Importing seaborn to make nice plots.
import seaborn as sns

Data_3 = pd.read_csv('Test_set_paper_for_testing.csv')
Data_3 = Data_3.values.tolist()


model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#
# # Loading the vector
Data_1 = pd.read_csv('claim_encode.csv')
Data_1 = Data_1.values.tolist()

Data_2 = pd.read_csv('gen_data.txt')
Data_2 = Data_2.values.tolist()
features = []
Y = []




for i in range(10000):
    temp = Data_1[i][0:768]
    features.append(Data_1[i][0:768])
    Y.append(0)


for i in range(5000):
    sentence = Data_3[i]
    sentence_embeddings = model.encode(sentence)
    features.append(sentence_embeddings[0])
    Y.append(2)

Y = np.array(Y)

digits_proj = TSNE(random_state=RS).fit_transform(features)


label = []

for i in Y:
    if(i == 0):
        label.append('Synthetic Data')
    elif(i == 1):
        label.append('Original Data')
    elif(i == 2):
        label.append('Test Data')

label = pd.DataFrame(label,columns =['Class'])

fig, ax = plt.subplots()

x = digits_proj[:,0]
y = digits_proj[:,1]

x = pd.DataFrame(x,columns =['x'])
y = pd.DataFrame(y,columns =['y'])

data = pd.concat([x,y,label],axis=1)
print(type(data))
print(data)
facet = sns.lmplot(data=data, x='x', y='y', hue='Class', fit_reg=False, legend=False, scatter_kws={"s": 1})

#add a legend
leg = facet.ax.legend(bbox_to_anchor=[1, 1],
                         title="Class", fancybox=False)
#change colors of labels
customPalette = ['#0c0107', '#0c0107']
for i, text in enumerate(leg.get_texts()):
    plt.setp(text, color = customPalette[i])

#plt.show()
plt.savefig('digits_tsne-generated_LaTextGAN.png', bbox_inches='tight')
