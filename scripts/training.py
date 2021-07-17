import random
import json
import joblib
import numpy as np
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('../intents.json').read())
# print(intents['intents'][0])
words = []
classes = []
documents = []
ignore_letters = ['?', '.', '!', ',', '\n']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # print(pattern)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # print(words)
        documents.append((word_list, intent['tag']))
        # print(documents)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            # print(classes)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# print(words)
words = sorted(set(words))
classes = sorted(set(classes))

joblib.dump(words, '../data/words.joblib')
joblib.dump(classes, '../data/classes.joblib')

training = []
output_empty = [0] * len(classes)
# print(len(output_empty)) // 25

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        # print(len(bag))

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    # print(training)


random.shuffle(training)
training = np.array(training, dtype=object)
# print(training)

x = list(training[:, 0])
y = list(training[:, 1])
# print(len(train_x), len(train_y))
train_x, train_y = x[:50], y[:50]
val_x, val_y = x[50:], y[50:]
print(len(train_x), len(train_y), len(val_x), len(val_y))
model = models.Sequential()
model.add(layers.Dense(500, input_shape=(len(train_x[0]),), activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="relu"))
# model.add(layers.InputLayer(input_shape=(len(train_x[0]),)))
model.add(layers.Dense(len(train_y[0]), activation="softmax"))

sgd = optimizers.SGD(0.01, momentum=0.9, nesterov=True)

early = callbacks.EarlyStopping(patience=200, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('../models/neural6.h5', save_best_only=True)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y),
          batch_size=5, epochs=500, verbose=1, validation_data=(np.array(val_x), np.array(val_y)),
          callbacks=[checkpoint, early])

# model.save('../models/model.h5')

print("Done")
