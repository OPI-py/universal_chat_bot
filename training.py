import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# itterate over intents
for intent in intents['intents']:
    # for each of  sub-value
    for pattern in intent['patterns']:
        # split text into individual words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # append word and it cattegory
        documents.append(((word_list), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# converting words to its base form inside list comprehension
words = [lemmatizer.lemmatize(word) for word  in words if word not in ignore_letters]
words = sorted(set(words)) # sort, eliminate duplicates

classes = sorted(set(classes))

# save words and classes into files as binaries
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes) # zero's template

# add documents data in the training list
for document in documents:
    bag = [] # for each of combination create empty bag of words
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # if word occur in pattern append 1 otherwise 0
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty) # copying list
    output_row[classes.index(document[1])] = 1 # set index in output_row to 1
    training.append([bag, output_row])

random.shuffle(training) # reorganize the order of the list
training = np.array(training)

# features and layers
train_x = list(training[:, 0]) # everything and zero dimension
train_y = list(training[:, 1])

# make Sequential model
model = Sequential()
# add entry level layer with 128 neurons and rectified linear unit activation function
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# add dropout layer with a frequency of rate at each step to prevent overfitting
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# end layer with as many neurons as there are training data elements for the labels
# activation scales the results in the output layer,shows probability distribution
model.add(Dense(len(train_y[0]), activation='softmax'))

# stochastic gradient descent optimizer
# lr - learning rate, decay=0.000001
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile and computes the crossentropy loss between the labels and predictions
# metrics - calculates how often predictions equal labels
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

ts_model = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.tf', ts_model)
print('Done')