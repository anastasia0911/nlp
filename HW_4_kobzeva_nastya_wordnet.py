import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk

print('1. Все значения (синсеты) для лексемы plant:')
plant_synsets = wordnet.synsets('plant')
for i in plant_synsets:
    print(str(i) + ': ' + i.definition())

print('\n2. Определения для лексемы plant в значении a)"завод" и b)"растение":')
print('a)' + wordnet.synset('plant.n.01').definition())
print('b)' + wordnet.synset('plant.n.02').definition())

''' Примеры употребления лексемы plant в этих значениях:
print(wordnet.synset('plant.n.01').examples())
>> [they built a large plant to manufacture automobiles]
print(wordnet.synset('plant.n.02').examples())
>> [] Почему-то не выдает примеров в этом значении '''

print('\n3. Алгоритм Леска для разрешения неоднозначности:')
s1 = 'They built a large plant to manufacture automobiles'
s2 = 'Aloe is a green plant that has long leaves with somewhat spiky edges'
print(s1, lesk(s1.split(), 'plant', 'n'))
print(s2, lesk(s2.split(), 'plant', 'n'))
# Без ограничения на pos этот алгоритм даже часть речи определяет неверно

print('\n4. Гиперонимы для значений a)"завод" и b)"растение":')
print('a)', wordnet.synset('plant.n.01').hypernyms())
print('b)', wordnet.synset('plant.n.02').hypernyms())

print('\n5. Наименьшее расстояние между значениями:')
industry_set = wordnet.synsets('industry')
plant_01 = wordnet.synset('plant.n.01')
d = {}
''' Вместо shortest_path_distance в этом пункте можно использовать обратную
величину - path_similarity, заменив min на max. В этом случае расстояние будет
в диапазоне от 0 до 1, причем чем ближе слова - тем ближе к 1. Я решила здесь
оставить shortest_path_distance, а в следующем пункте воспользоваться path_
similarity, хотя по сути это одно и то же. '''
for i in range(len(industry_set)):
    dist = plant_01.shortest_path_distance(industry_set[i])
    if industry_set[i] not in d:
        d[industry_set[i]] = dist
min_el = min([d[i] for i in d if d[i] is not None])
for k in d:
    if d[k] == min_el:
        print('a)', k, min_el)

leaf_set = wordnet.synsets('leaf')
plant_02 = wordnet.synset('plant.n.02')
d = {}
for i in range(len(leaf_set)):
    dist = plant_02.shortest_path_distance(leaf_set[i])
    if leaf_set[i] not in d:
        d[leaf_set[i]] = dist
min_el = min([d[i] for i in d if d[i] is not None])
for k in d:
    if d[k] == min_el:
        print('b)', k, min_el)

print('\n6. Вычисление расстояния двумя разными способами:')
rtl = wordnet.synsets("rattlesnake's master")
# print(rtl)
# Этого слова нет в Wordnet'e, я решила взять другое
print('\nPlant and petroselinum')
plant = wordnet.synset('plant.n.02')
parsley = wordnet.synsets('petroselinum')
# для petroselinum есть всего один синсет
print('path_similarity', plant.path_similarity(parsley[0]))
''' Вычисление сходства с использованием величины, которая прямо пропорциональна
глубине, на которой находится общий предок: '''
print('wup_similarity', plant.wup_similarity(parsley[0]))

print('\nOrganism and whole')
organism = wordnet.synsets('organism')
# Ограничение на pos, т.к. сравнивать можно только слова одной части речи
whole = wordnet.synsets('whole', 'n')
for i in range(len(organism)):
    for k in range(len(whole)):
        sim1 = organism[i].path_similarity(whole[k])
        sim2 = organism[i].wup_similarity(whole[k])
        print('path_similarity', organism[i], whole[k], sim1)
        print('wup_similarity', organism[i], whole[k], sim2)
''' На мой взгляд, wup_similarity лучше отражает интуитивное представление о
семантической близости слов, потому что эта метрика использует для сравнения
глубину общего предка двух слов и лучше отражает расстояние между более общими
и более частными синсетами. '''

