import pickle
import  numpy as np
"""
this is the module that will be slightly different for me 
I will be using special bit block box that maps certain words into same bit block
"""

def text_to_tensor(filePath):
    """
        Read text from file
    """
    with open(filePath, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        print(lines)
    f.close()
    corpus = []
    for l in lines:
        l = l.strip().split(' ') #strip removes blank spaces from both sides
        if len(l) < 40:
            corpus.append(l)
    """
    Get all words used in text
    """
    vocab = []
    print("Here 1")
    for p in corpus:
        vocab.extend(p) #save all into a single list
    vocab = list(set(vocab)) #save only unique characters

    genData = np.load('data/gen_corpus.npy')

    for genD in genData:
        tempList = []
        for data in genD:
            temp = vocab[data-1]
            tempList.append(temp)
        sent = ' '.join(tempList)
        print(sent)


"""
    Convert text data to numbers 
"""

text_to_tensor('Data_LeakGAN.txt')