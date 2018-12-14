from flask import Flask
from flask import request, jsonify
import start_inference

app = Flask(__name__)


def convert(answers_scores):
    best_answer, best_score = answers_scores[0]
    scores = [score for answer, score in answers_scores]
    average = sum(scores)/len(scores)
    results = [{"answer": answer, "score": score} for answer, score in answers_scores]
    return {"results": results, "best_answer": best_answer, "best_index": 0, "best_score": best_score, "average_score": average}


@app.route("/question", methods=['GET'])
def get_answer():

    question = request.args.get("question")

    print("question: ", question)
    answers_scores = start_inference.do_inference(question)
    answer = convert(answers_scores)
    return jsonify(answer)


@app.route("/question", methods=['POST'])
def answer_post():
    question = request.form.get('question')

    if not question:
        question = request.json.get('question')
    print("question: ", question)
    answer = convert(start_inference.do_inference(question))
    return jsonify(answer)


if __name__ == "__main__":
  app.run(host='127.0.0.1', port=8080)
