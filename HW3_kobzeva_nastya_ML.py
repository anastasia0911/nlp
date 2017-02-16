''' Features:
длина предложения в буквах
число различных букв в предложении
число гласных в предложении
медиана числа букв в слове
медиана числа гласных в слове '''

import re
import pymystem3
import numpy as np
from matplotlib import pyplot as plt

with open('anna.txt', encoding='utf-8') as f:
    anna = f.read()
with open('sonets.txt', encoding='utf-8') as f:
    sonets = f.read()

anna_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', anna)
sonet_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', sonets)
# print(None in anna_sentences)
# print(len(anna_sentences), len(sonet_sentences))

def lenwords(sentence):
    return [len(word) for word in sentence.split()]

anna_sentlens = [lenwords(sentence) for sentence in anna_sentences
                 if len(lenwords(sentence)) > 0]
sonet_sentlens = [lenwords(sentence) for sentence in sonet_sentences
                  if len(lenwords(sentence)) > 0]
anna_data = [(len(sentence), np.mean(sentence),
              np.median(sentence), np.std(sentence)) 
             for sentence in anna_sentlens]
sonet_data = [(len(sentence), np.mean(sentence), np.median(sentence),
               np.std(sentence)) for sentence in sonet_sentlens]

# print(anna_data[:10])

anna_data = np.array(anna_data)
sonet_data = np.array(sonet_data)
plt.figure()
plt.plot(anna_data[:,0], anna_data[:,3], 'o')
plt.show()

plt.figure()
c1, c2 = 0, 1
plt.plot(anna_data[:,c1], anna_data[:,c2], 'og', 
         sonet_data[:,c1], sonet_data[:,c2], 'sb')
plt.show()
