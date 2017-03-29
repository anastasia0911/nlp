''' Features:
длина предложения в буквах
число различных букв в предложении
число гласных в предложении
медиана числа букв в слове
медиана числа гласных в слове '''

import re
import numpy as np
from matplotlib import mlab
import time
from pymystem3 import Mystem
from sklearn import model_selection, svm
from matplotlib import pyplot as plt

with open('anna.txt', encoding='utf-8') as f:
    anna = f.read()
with open('sonets.txt', encoding='utf-8') as f:
    sonets = f.read()

anna_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', anna)
sonet_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', sonets)

def lensent(sentence):
    return [len(word) for word in sentence.split()]

# число уникальных букв в предложении
def uni_let(sent):
    letters = set()
    for word in sent:
        word = word.strip(',.?^&*()-_#{}[]\\—<>!@"\';:/%')
        for i in word: 
            if i not in letters:
                letters.add(i)
    return len(letters)

# число гласных в слове
def vowel(word):
    vowels = 0
    for i in word:
        if i in 'уеыаоэёяию':
            vowels += 1
    return vowels

def get_features(label, corpus):
    features = [] # label - анна или сонеты
    for line in corpus:
        line = line.strip().lower()
        line = line.split(' ')
        if len(line) > 0:
            uni_letters = uni_let(line)
            letters = []
            vowels = []
            for word in line:
                if len(word) > 0:
                    word = word.strip(',.?^&*()-_#{}[]\\—<>!@"\';:/%')
                    letters.append(len(word))
                    vowels.append(vowel(word))
            if len(letters) > 0: 
                features.append([label, np.sum(letters), uni_letters,
                                 np.sum(vowels), np.median(letters),
                                 np.median(vowels)])
    return features



anna_data = get_features(1, anna_sentences)
sonet_data = get_features(2, sonet_sentences)
anna_data = np.array(anna_data)
sonet_data = np.array(sonet_data)
data = np.vstack((anna_data, sonet_data))

p = mlab.PCA(data[:, 1:], True)
N = len(anna_data)
plt.figure()
plt.plot(p.Y[:N,0], p.Y[:N,1], 'og', p.Y[N:,0], p.Y[N:,1], 'sb')
plt.show()

clf = svm.LinearSVC(C=0.1)
clf.fit(data[::2, 1:], data[::2, 0])
print(clf.score(data[::2, 1:], data[::2, 0])) # 0.93, неплохо!

wrong = 0
for obj in data[1::2, :]:
    label = clf.predict(obj[1:].reshape(1, -1))
    if label != obj[0] and wrong < 3:
        print('Пример ошибки машины: class = ', obj[0],
              ', label = ', label, ', экземпляр ', obj[1:])
        wrong += 1
    if wrong > 3:
        break
