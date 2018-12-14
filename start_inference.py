from nmt.nmt.nmt import import_hparams
from nmt.nmt.inference import initialize, get_answers
from scorer import Scorer
from nltk.tokenize import word_tokenize
from improve import improve_answer
from settings import score

scorer = Scorer(score)
model_path = score['model_path']

hparams = import_hparams(model_path)
hparams.infer_mode = "beam_search"
hparams.beam_width = 10
hparams.num_translations_per_input = 10

initialize(hparams)


def get_sorted_answers(answers):
    return sorted(answers, key=lambda x: x[1], reverse=True)


def do_inference(question):
    question = question.strip()
    if not (question.endswith('.') or question.endswith('!') or question.endswith('?')):
        question = question + '?'
    question = ' '.join(word_tokenize(question.lower()))
    answers = get_answers(question)
    scores = scorer.get_score(question, answers)
    improved_answers = [improve_answer(answer) for answer in answers]
    answers_scores = zip(improved_answers, scores)
    return list(get_sorted_answers(answers_scores))
