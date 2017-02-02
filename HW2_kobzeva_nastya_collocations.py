# coding=utf-8
# encoding='utf-8-sig'
import nltk
from nltk.collocations import *
from nltk.metrics.spearman import *

bigram_measures = nltk.collocations.BigramAssocMeasures()


def get_bigrams(filename):
    lines = open(filename, encoding='utf-8-sig').readlines()
    new_lines = []
    for line in lines:
        ''' Избавляюсь от лишних пробелов и запятых, удаляю колонку со
        словом суд, чтобы не искать триграммы. '''
        new_line = [word.strip(',') for word in line.split() if len(word) > 1]
        del new_line[0]
        new_lines.append(new_line)
    return new_lines


def comparison(array, measure):
    finder = BigramCollocationFinder.from_documents(array)
    ''' Золотой стандарт, биграммы в котором отсортированы по убыванию.
    Я сделала его ручками, просто посмотрев на частые сочетания в табличке.'''
    gold_standard = [('УДОВЛЕТВОРИТЬ', 'ИСК'), ('ПРИНЯТЬ', 'РЕШЕНИЕ'),
                     ('УДОВЛЕТВОРИТЬ', 'ХОДАТАЙСТВО'), ('ВЫНЕСТИ', 'РЕШЕНИЕ'),
                     ('ВЫДАТЬ', 'САНКЦИЯ'), ('ПРИЗНАТЬ', 'ВИНОВНАЯ'),
                     ('НАЛОЖИТЬ', 'АРЕСТ'), ('САНКЦИОНИРОВАТЬ', 'АРЕСТ'),
                     ('ОТКЛОНИТЬ', 'ИСК'), ('РАССМОТРЕТЬ', 'ИСК')]

    ''' Ищу биграммы из золотого стандарта в полученном списке биграмм и
    кладу их в отдельный массив вместе со значениями. '''
    scored = finder.score_ngrams(measure)
    scored_gs = []
    for i in scored:
        for b in gold_standard:
            if i[0][0] == b[0] and i[0][1] == b[1]:
                scored_gs.append(i)
    ''' Сортирую биграммы из золотого стандарта по убыванию значений метрики.
    Записываю в новый массив биграммы в нужном порядке. '''
    results = []
    for i in sorted(scored_gs, key=lambda bigram: bigram[-1], reverse=True):
        results.append(i[0])
    # Сравниваю золотой стандарт с полученным массивом.
    print('Сравнение золотого стандарта со списком биграмм,'
          'полученных при помощи ' + str(measure))
    print('%0.1f' % spearman_correlation(ranks_from_sequence(gold_standard),
                                         ranks_from_sequence(results)))
    ''' Сначала я подумала, что нужно сравнить с золотым стандартом
    10 лучших результатов. Не знаю, насколько это осмысленно, но я делала
    это так. '''
    best_results = finder.nbest(measure, 10)
    # print('Top-10 биграмм, полученных при помощи ' + str(measure))
    # print(best_results)
    print('Сравнение золотого стандарта с top-10 по этому методу')
    print('%0.1f' % spearman_correlation(ranks_from_sequence(gold_standard),
                                         ranks_from_sequence(best_results)))


def main():
    array = get_bigrams('court-V-N.csv')
    likelihood = bigram_measures.likelihood_ratio
    student_t = bigram_measures.student_t
    comparison(array, likelihood)
    comparison(array, student_t)


if __name__ == '__main__':
    main()


''' Таким образом, с помощью метрики likelihood ratio я получила ранжированный
список биграмм, в котором биграммы из золотого стандарта были распределены
почти так же, как я ранжировала их в золотом стандарте. Коэффициент Спирмена
получился равным 0.7, что свидетельствует о сильной прямой связи между
величинами. С помощью t-критерия Стьюдента ранжированный список биграмм из
золотого стандарта получился точно таким же, как и сам золотой стандарт, ро
Спирмена = 1, строгая прямая связь. При этом при сравнении топ-10 результатов
t-критерия с золотым стандартом коэффициент Спирмена так же получился очень
высоким - 09. '''
