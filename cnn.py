import logging
import time
from platform import python_version
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

#filename = 'Data_2500.csv'

df = pd.read_csv('data/Data.csv')
df_test = pd.read_csv('data/Test_set_paper_for_testing.csv')
df_test = df

df_train = df.reset_index(drop=True)
df_val = df.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

model_class = transformers.BertModel
tokenizer_class = transformers.BertTokenizer
pretrained_weights='bert-base-uncased'
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)

print("model loading done")

max_seq = 100


def tokenize_text(df, max_seq):
    return [
        tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.claims.values
    ]

def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])

def tokenize_and_pad_text(df, max_seq):
    tokenized_text = tokenize_text(df, max_seq)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)

def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)



train_indices = tokenize_and_pad_text(df_train, max_seq)
train_indices = train_indices.type(torch.LongTensor)

val_indices = tokenize_and_pad_text(df_val, max_seq)
val_indices = val_indices.type(torch.LongTensor)

test_indices = tokenize_and_pad_text(df_test, max_seq)
test_indices = test_indices.type(torch.LongTensor)

target_columns = ['classes']


with torch.no_grad():
    x_train = bert_model(train_indices)[0]
    x_val = bert_model(val_indices)[0]
    x_test = bert_model(test_indices)[0]


y_train = targets_to_tensor(df_train, target_columns)
y_val = targets_to_tensor(df_val, target_columns)
y_test = targets_to_tensor(df_test, target_columns)

print("Tokenizing data")
class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(KimCNN, self).__init__()

        V = embed_num
        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes

        self.static = static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.softmax = nn.Softmax()

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)


        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.softmax(logit)
        return output

embed_num = x_train.shape[1]
embed_dim = x_train.shape[2]
class_num = y_train.shape[1]
kernel_num = 5
kernel_sizes = [2, 3, 4, 5, 6]
dropout = 0.5
static = True


model = KimCNN(
    embed_num=embed_num,
    embed_dim=embed_dim,
    class_num=class_num,
    kernel_num=kernel_num,
    kernel_sizes=kernel_sizes,
    dropout=dropout,
    static=static,
)
print("Model created")

n_epochs = 1
batch_size = 10
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()


def generate_batch_data(x, y, batch_size):
    i, batch = 0, 0
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield x_batch, y_batch, batch
    if i + batch_size < len(x):
        yield x[i + batch_size :], y[i + batch_size :], batch + 1
    if batch == 0:
        yield x, y, 1


train_losses, val_losses = [], []

for epoch in range(n_epochs):
    print("Epoch number:",epoch)
    start_time = time.time()
    train_loss = 0

    model.train(True)
    model = nn.DataParallel(model)
    #model.to(f'cuda:{model.device_ids[0]}')


    for x_batch, y_batch, batch in generate_batch_data(x_train, y_train, batch_size):

        x_batch = x_batch.to(f'cuda:{model.device_ids[0]}')
        y_batch = y_batch.to(f'cuda:{model.device_ids[0]}')

        model.to(f'cuda:{model.device_ids[0]}')
        y_pred = model(x_batch)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= batch
    train_losses.append(train_loss)
    elapsed = time.time() - start_time
    #model = nn.DataParallel(model)
    model.eval() # disable dropout for deterministic output
    #model.to(f'cuda:{model.device_ids[0]}')
    with torch.no_grad(): # deactivate autograd engine to reduce memory usage and speed up computations
        val_loss, batch = 0, 1
        for x_batch, y_batch, batch in generate_batch_data(x_val, y_val, batch_size):



            x_batch = x_batch.to(f'cuda:{model.device_ids[0]}')
            y_batch = y_batch.to(f'cuda:{model.device_ids[0]}')
            #model.to(f'cuda:{model.device_ids[0]}')
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
        val_loss /= batch
        val_losses.append(val_loss)

    print(
        "Epoch %d Train loss: %.2f. Validation loss: %.2f. Elapsed time: %.2fs."
        % (epoch + 1, train_losses[-1], val_losses[-1], elapsed)
    )

torch.save(model.state_dict(), "data/BERT_CNN_Full_With_GAN.model")

model.eval() # disable dropout for deterministic output
with torch.no_grad(): # deactivate autograd engine to reduce memory usage and speed up computations
    y_preds = []
    batch = 0
    for x_batch, y_batch, batch in generate_batch_data(x_test, y_test, batch_size):
        y_pred = model(x_batch)
        y_preds.extend(y_pred.cpu().numpy().tolist())
    y_preds_np = np.array(y_preds)

y_test_np = df_test[target_columns].values


precision_macro = precision_score(y_test_np, y_preds_np, average='macro')
recall_macro = recall_score(y_test_np, y_preds_np, average='macro')
f1_macro = f1_score(y_test_np, y_preds_np, average='macro')

print("precision_macro:",precision_macro)
print("recall_macro:",recall_macro)
print("f1_macro:",f1_macro)


precision_micro = precision_score(y_test_np, y_preds_np, average='micro')
recall_micro = recall_score(y_test_np, y_preds_np, average='micro')
f1_micro = f1_score(y_test_np, y_preds_np, average='micro')

print("precision_micro:",precision_micro)
print("recall_micro:",recall_micro)
print("f1_micro:",f1_micro)

with open('result_cnn_bert.csv', mode='a') as f:
    f.write('\n')
    f.write('BERT_CNN')
    f.write('\n')
    f.write("precision_macro:" + str(precision_macro))
    f.write(" recall_macro: " + str(recall_macro))
    f.write(" f1_macro: " + str(f1_macro))
    f.write('\n')
    f.write("precision_micro:" + str(precision_micro))
    f.write(" recall_micro: " + str(recall_micro))
    f.write(" f1_micro: " + str(f1_micro))

f.close()