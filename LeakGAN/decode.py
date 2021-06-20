import pickle
import numpy as np
import pandas as pd
data_Name = "cotra"
generated_file = "data/generated/gen_corpus.npy"
output_file = "data/generated/out_put.text"
vocab_file = pickle.load(open('data/chars.pkl', 'rb'))
generated_data = np.load(generated_file)

#print(vocab_file)
#file_out = open("data/generated/gen_data.txt","w+")
sentence_list = []
for i in range(len(generated_data)):
    sentence = ""
    for j in range(len(generated_data[i])):
        number = generated_data[i][j]
        chr = vocab_file[number]
        sentence = sentence + chr + " "
    #print(sentence)
    sentence_list.append(sentence)
    #file_out.write(sentence)  # .encode('utf-8'))
    #print("\n\n")

sentence_list = pd.DataFrame(sentence_list)
sentence_list.to_csv("data/generated/gen_data.txt", index=False)
#word, vocab = pickle.load(open('save/'+vocab_file))
# print (len(word))
# input_file = 'save/generator_sample.txt'
# # input_file = 'save/coco_451.txt'
# output_file = 'speech/' + data_Name + '_' + input_file.split('_')[-1]
# with open(output_file, 'w')as fout:
#     with open(input_file)as fin:
#         for line in fin:
#             #line.decode('utf-8')
#             line = line.split()
#             #line.pop()
#             #line.pop()
#             line = [int(x) for x in line]
#             line = [word[x] for x in line]
#             # if 'OTHERPAD' not in line:
#             line = ' '.join(line) + '\n'
#             fout.write(line)#.encode('utf-8'))
