import random
import json
import joblib
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras import models

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('../intents.json').read())

words = joblib.load('../data/words.joblib')
classes = joblib.load('../data/classes.joblib')
model = models.load_model('../models/neural4.h5')


def cleanup(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = cleanup(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_classes(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    # print(return_list)
    return return_list


def get_res(intents_list, intent_json):
    # print("Intent list", intents_list)
    result = ""
    tag = intents_list[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


while True:
    message = input("")
    ints = predict_classes(message)
    res = get_res(ints, intents)
    print("Chatly:) ", res)
