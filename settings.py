import os
chatbot_path = os.getcwd()
score = {
    "model_path": "./nmt_model/luong5embed",
    "unk_modifier": -100,
    "ending_modifier": 10,
    "length_modifier": 3,
    "phrase_modifier": 4,
    "similarity_modifier": 3,
    "yes_no_modifier": 2,
    "repetition_modifier": 1,
    "vocab_file": "./nmt_data/vocab20k.to",
    "penalty_phrases": ['i don t know', 'i m not sure', 'i don t think', "i don t remember"]
}