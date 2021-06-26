import numpy as np
import pickle
import argparse
import sys
import random
import pandas as pd

def load_sents():

    with open("wiki.10million.raw.txt", "r") as f:
        lines = f.readlines()
    lines =  [l.strip() for l in lines]
    return lines
   
    
    
def get_word2bin(lines, num_bins):

    w2bin = {}
    with open("w2i.pickle", "rb") as f:
        data = pickle.load(f)
    
    words_by_freq = list(data.keys())
    bin_size = int(len(words_by_freq) / num_bins)
    for i in range(num_bins):
    
        word_bin = words_by_freq[i * bin_size : (i + 1) * bin_size]
        for w in word_bin:
            w2bin[w] = i
    
    return w2bin
       

def generate_data(lines, w2bin, size, window_size, train=True):

    l = len(lines)
    relevant = lines[:int(l/2)] if train else lines[int(l/2):]
    data = []
    chosen = set()
    
    while len(data) < size:
    
        i = random.choice(range(len(relevant)))
        sent = relevant[i].lower().split(" ")
        for k,w in enumerate(sent):
            if w not in w2bin:
            
                sent[k] = "[UNK]"
        
        j = -1
        q = 0
        chose = False
        
        while q < 30:
            q += 1
            j = random.choice(range(len(sent)))
            if ((i,j) not in chosen) and (sent[j] != "[UNK]"): 
                chosen.add((i,j))
                chose = True
                break
           
        if chose:
        
            example = sent[:]
            example[j] = "[MASK]"
            example = example[max(0, j - window_size): min(len(example), j + window_size)]

            mask_ind = example.index("[MASK]")
            left_cxt_size = len(example[:mask_ind])
            right_cxt_size = len(example[mask_ind:])
            example = ["<S>"] * (window_size - left_cxt_size) + example + ["<E>"] * (window_size - right_cxt_size+1)
            
            example_str = " ".join(example)
            masked_word = sent[j]
            masked_bin = w2bin[masked_word]
            data.append({"sent_ind": i, "word_index": j, "example": example, "word": masked_word, "freq": masked_bin})
    return data
        

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='generate data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-size', dest='train_size', type=int,
                        default=8000000)
    parser.add_argument('--dev-size', dest='dev_size', type=int,
                        default=8000000) 
    parser.add_argument('--freq-bins', dest='freq_bins', type=int,
                        default=10)
    parser.add_argument('--window-size', dest='window_size', type=int,
                        default=5)                                                                            
    args = parser.parse_args()
    
    sents = load_sents()
    w2bin = get_word2bin(sents, args.freq_bins)
    
    train_data = generate_data(sents, w2bin, args.train_size, args.window_size, train=True)
    dev_data = generate_data(sents, w2bin, args.dev_size, args.window_size, train=False)
    
    train_df = pd.DataFrame(train_data)
    del train_data
    train_df.to_csv("train.tsv", sep="\t")
    del train_df
    dev_df = pd.DataFrame(dev_data)
    del dev_data
    dev_df.to_csv("dev.tsv", sep = "\t")
