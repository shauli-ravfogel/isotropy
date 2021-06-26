from collections import Counter
import pickle
import numpy as np
import tqdm


with open("wiki.10million.raw.txt", "r") as f:
    lines = f.readlines()
    
counter = Counter()
for line in tqdm.tqdm(lines):
    counter.update(line.lower().strip().split(" "))
    
top = counter.most_common(35000)
words, counts = list(zip(*top))
with open("vocab.txt", "w") as f:
    for w,c in top:
        f.write(w + "\t" + str(c) + "\n")
        
w2i = {w:i for i,w in enumerate(words)}
i2w = {i:w for i,w in enumerate(words)}

for i,w in enumerate(["[UNK]", "[MASK]", "<S>", "<E>"]):

    w2i[w] = len(words) + i 
    i2w[len(words) + i ] = w

with open("w2i.pickle", "wb") as f:
    pickle.dump(w2i, f)
  
with open("i2w.pickle", "wb") as f:
    pickle.dump(i2w, f)  
