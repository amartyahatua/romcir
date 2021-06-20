import pickle
import numpy as np

# dataSet = np.load('data/gen_corpus.npy')
# corpus = np.load('data/corpus.npy')
# print(corpus)
vocab = pickle.load("data/chars.pkl" )
print(vocab)
output_file = 'data/output_gen'

with open(output_file, 'w')as fout:
    for data in dataSet:
        temp = []
        for i in data:
            if(i < max(corpus)):
                wrd = word[i - 1]
                print(wrd)
        line = [word[x-1] for x in temp]
        line = ' '.join(line)
        print(line)




    with open(input_file)as fin:
        for line in fin:
            line = [word[x] for x in line]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)#.encode('utf-8'))

