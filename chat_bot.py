import random
import json
import pickle
import numpy as np
import datetime

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# load words and classes models
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbotmodel.tf')

def clean_up_sentence(sentence):
    ''' Tokenize, lematize, return words from numerical data'''
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    '''Convert  sentence into list of zeros and ones that indicates if
     word is there or not'''
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) # as many zeros as words
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:  # if word inside bag - assign 1
                bag[i] = 1
    return np.array(bag)
    
def predict_class(sentence):
    '''Get and return prediction'''
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]  # predict bag of words
    ERROR_THRESHOLD = 0.25
    # enumerate all results, get index class and probability
    # return results is prediction larger than ERROR_THRESHOLD
    results = [[i, r] for i, r  in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)# highest probability first
    return_list = []
    for r in results:
        # return actual result of classes and probabilities
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
    
def get_response(intents_list, intents_json):
    '''Return response'''
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == 'current_time':
            result = i['responses'][0]
            break
        elif i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def show_response():
    if res.startswith('print'):
        return exec(res)
    else:
        print('##', res)

print('Bot is running! :)')

exit_command = ['exit', 'quit', 'leave']
while True:
    message = input('>> ')
    ints = predict_class(message)
    res = get_response(ints, intents)
    if message.lower() in exit_command:
        print('## Farewell!')
        break
    show_response()