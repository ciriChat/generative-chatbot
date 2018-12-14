from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

apo_words = set()
with open('apostrophes.txt', 'r', encoding='utf8') as input_f:
    lines = input_f.readlines()
    for line in lines:
        apo_words.add(line.strip())


def remove_spaces(answer):
    for sign in ["'", ".", ",", ":"]:
        exp = " " + sign
        answer = answer.replace(exp, sign)
    return answer.replace("' ", "'")


def truecase(answer):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(answer)
    improved_tokens = []
    for token in tokens:
        pos_token_lower = pos_tag([token])
        pos_token = pos_tag([token.capitalize()])
        word_lower, pos_lower = pos_token_lower[0]
        word, pos = pos_token[0]
        if (pos_lower == 'NN' and (pos == 'NNP' or pos == 'NNPS')) or word == 'I':
            improved_tokens.append(word)
        else:
            improved_tokens.append(token)
    return ' '.join(improved_tokens)


def fix_apostrophes(answer):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(answer)
    new_answer = []
    i = 0
    max_len = len(tokens)-1
    while i < max_len:
        first = tokens[i]
        second = tokens[i+1]
        candidate = "'".join([first, second])
        if candidate.lower() in apo_words:
            new_answer.append(candidate)
            i += 1
        else:
            new_answer.append(first)
        i += 1
    new_answer.append(tokens[-1])
    return ' '.join(new_answer)


def improve_answer(answer):
    answer = fix_apostrophes(answer)
    answer = answer.capitalize()
    answer = truecase(answer)
    answer = remove_spaces(answer)
    return answer
