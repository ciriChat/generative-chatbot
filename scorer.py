import Levenshtein
from collections import Counter
from nltk.tokenize import word_tokenize
from math import log2


class Scorer:

    def __init__(self, settings):
        self.unk_modifier = settings['unk_modifier']
        self.ending_modifier = settings['ending_modifier']
        self.length_modifier = settings['length_modifier']
        self.penalty_phrases = settings['penalty_phrases']
        self.phrase_modifier = settings['phrase_modifier']
        self.similarity_modifier = settings['similarity_modifier']
        self.yes_no_modifier = settings['yes_no_modifier']
        self.repetition_modifier = settings['repetition_modifier']
        self.words_values = dict()
        with open(settings['vocab_file'], 'r', encoding='utf8') as input_file:
            for i, line in enumerate(input_file, 1):
                self.words_values[line.strip()] = i

    def _unk_score(self, answer):
        unk_ratio = answer.count('unk')/(len(word_tokenize(answer)))
        score = (self.unk_modifier*unk_ratio)
        return score

    def _ending_score(self, answer):
        if answer.strip().endswith('.'):
            score = self.ending_modifier/2
        else:
            score = -self.ending_modifier
        return score

    def _filter_answer(self, answer):
        for phrase in self.penalty_phrases:
            answer = answer.replace(phrase, ' ')
        return answer

    def _length_score(self, answer):
        clean_answer = self._filter_answer(answer)
        score = self.length_modifier*log2(max(len(word_tokenize(clean_answer)), 1))
        return score

    def _bad_phrases_score(self, answer):
        counter = 0
        for phrase in self.penalty_phrases:
            if phrase in answer:
                counter += 1
        score = (-counter*self.phrase_modifier)
        return score

    def _word_value(self, answer):
        tokens = word_tokenize(answer)
        score = 0
        for token in tokens:
            if token in self.words_values:
                val = self.words_values[token]
                if val < 100:
                    continue
                else:
                    score += log2(val)

        return score/len(tokens)

    def _similarity_score(self, answer, question):
        distance = Levenshtein.distance(question, answer)/max(len(question), len(answer))
        if distance <= 0.2:
            return -self.similarity_modifier
        else:
            return (distance-0.5) * self.similarity_modifier

    def _yes_no_score(self, answer):
        answer = word_tokenize(answer)
        sum = 0
        for y in ['yes', 'yeah', 'you', 'right']:
            if y in answer:
                sum += self.yes_no_modifier
        for n in ['no', 'sir']:
            if n in answer:
                sum -= self.yes_no_modifier
        return sum

    def _repetitions_score(self, answer):
        cnt = Counter()
        for word in word_tokenize(answer):
            cnt[word] += 1
        number = cnt.most_common(1)[0][1]
        if number <= 2:
            number = number/2
        return -number*self.repetition_modifier

    def _index_score(self, index):
        return 4/(1 + index)

    def get_score(self, question, answers):
        scores = []
        answer_funcs = [self._unk_score, self._ending_score, self._length_score, self._bad_phrases_score, self._yes_no_score, self._repetitions_score, self._word_value]
        answer_question_funcs = []  # [_similarity_score]

        index_funcs = [self._index_score]

        for index, answer in enumerate(answers):
            answer_score = 0.0
            for fun in answer_funcs:
                answer_score += fun(answer)
            for fun in answer_question_funcs:
                answer_score += fun(answer, question)
            for fun in index_funcs:
                answer_score += fun(index)
            scores.append(answer_score)
        return scores

