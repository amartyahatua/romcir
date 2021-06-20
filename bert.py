import torch
from tqdm.notebook import tqdm
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import random
from transformers import BertForSequenceClassification
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import confusion_matrix



device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def BERTClassifier(df):


    df.head()

    df['classes'].value_counts()
    #
    #
    possible_labels = df.classes.unique()
    #
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)

    df['label'] = df.classes.replace(label_dict)
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                      df.label.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=df.label.values)

    df['data_type'] = ['not_set']*df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    print(df.groupby(['classes', 'label', 'data_type']).count())
    #
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    #
    encoded_data_train = tokenizer.batch_encode_plus(
        df.claims.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    df_valid = pd.read_csv('data/Test_set_paper_for_testing.csv')
    df_valid['label'] = df_valid.classes.replace(label_dict)

    encoded_data_val = tokenizer.batch_encode_plus(
        df_valid.claims.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df.label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_valid.label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    #model = nn.DataParallel(model)


    batch_size = 10

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)

    from transformers import AdamW, get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    epochs = 3

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    from sklearn.metrics import f1_score


    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')


    def accuracy_per_class(preds, labels):
        label_dict_inverse = {v: k for k, v in label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for i in range(len(preds_flat)):
            if(preds_flat[i] == 0):
                preds_flat[i] = -1
            elif(preds_flat[i] == 2):
                preds_flat[i] = 0


        precision = precision_score(labels_flat, preds_flat, average='macro')
        recall = recall_score(labels_flat, preds_flat, average='macro')
        f1 = f1_score(labels_flat, preds_flat, average='macro')
        cm = confusion_matrix(labels_flat, preds_flat)

        precision_micro = precision_score(labels_flat, preds_flat, average='micro')
        recall_micro = recall_score(labels_flat, preds_flat, average='micro')
        f1_micro = f1_score(labels_flat, preds_flat, average='micro')


        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')

        return precision, recall, f1, cm, precision_micro, recall_micro, f1_micro


    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    def evaluate(dataloader_val):

        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                model.to(device)
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals


    for epoch in tqdm(range(1, epochs + 1)):
    
        model.train()
        #model = nn.DataParallel(model)
    
        loss_train_total = 0
    
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            #model = nn.DataParallel(model)
    
            batch = tuple(b.to(device) for b in batch)
    
    
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }
    
            #inputs.to(device)
            model.to(device)
    
            outputs = model(**inputs)
            #model = nn.DataParallel(model)
    
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            optimizer.step()
            scheduler.step()
    
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
    
        torch.save(model.state_dict(), f'data/bertIterative/finetuned_BERT_epoch_{epoch}.model')
    
        tqdm.write(f'\nEpoch {epoch}')
    
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
    
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')


    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    model.to(device)
    model = nn.DataParallel(model)

    arr = os.listdir('data/bertIterative')
    f1_list = []
    precision_list =[]
    recall_list = []


    for filename in arr:
        model.load_state_dict(torch.load('data/bertIterative/'+str(filename)))
        #model.load_state_dict(torch.load('claim_evidence/68833_training_models/'+str(filename)))
        _, predictions, true_vals = evaluate(dataloader_validation)



        precision, recall, f1, cm, precision_micro, recall_micro, f1_micro = accuracy_per_class(predictions, true_vals)
        print(cm)
        #tn, fp, fn, tp = cm
        with open('result.csv', mode='a') as f:
            f.write('\n')

        className = pd.DataFrame(["NEI", "Refute", "Supported"])
        experiment_name = []
        experiment_name.append(str(filename))



        with open('result.csv', mode='a') as f:
            f.write('\n')
            f.write(str(filename))
            f.write("Precision:"+str(precision))
            f.write("Recall: "+str(recall))
            f.write("F1 score: "+str(f1))
            f.write("CM score: " + str(cm))

            f.write("Precision micro : " + str(precision_micro))
            f.write("  Recall micro : " + str(recall_micro))
            f.write("  F1 score micro: " + str(f1_micro))

            f.write('\n')

            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)


        f.close()

        print("Done"+str(filename))



df = pd.read_csv('data/Data.csv')
df_new = df.iloc[0:793414]

df_p = df_new[df_new['classes']==1]
df_n = df_new[df_new['classes']==-1]
df_i = df_new[df_new['classes']==0]

df_p = df_p.iloc[0:21408]
df_n = df_n.iloc[0:8578]
df_i = df_i.iloc[0:8990]

original = pd.concat([df_p,df_n,df_i], axis=0)


half = 9983
total = 2*half
percent = 100
increment = 5
#original = df

ganData = None
ganDataStarts = 793415
for i in range(5,10,5):
    ganPer = i
    ganDataCount = int((total*ganPer)/100)
    ganDataEnds = ganDataStarts + ganDataCount
    ganData = df.iloc[ganDataStarts:ganDataEnds]
    trainingData = pd.concat([original, ganData], axis=0)
    print(trainingData.head())
    print("***********************")
    print("Percent of GAN data =",i)
    print("Percent of Original data =", (100-i))
    print(len(trainingData))
    BERTClassifier(trainingData)