from flask import Flask, render_template,request, request,url_for
from flask_bootstrap import Bootstrap 


# NLP Packages
import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

app = Flask(__name__)
Bootstrap(app)

@app.before_first_request
def init():
    global model, word_index
    model  = keras.models.load_model('classify-1.h5')

    data = keras.datasets.imdb

    word_index = data.get_word_index()
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

def review_encode(s):
    encode = [1]

    for word in s:
        if word in word_index:
            encode.append(word_index[word.lower()])
        else:
            encode.append(2)

    return encode

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE," ",text)
    text = re.sub(BAD_SYMBOLS_RE,"",text)
    text = text.split()
    return ' '.join([i for i in text])

@app.route('/')
def index():
	return render_template('sentiments.html')

@app.route('/sentiments', methods = ['POST', "GET"])
def sentiments():
    if request.method == 'POST':
        line = request.form['rawtext']
        nline = text_prepare(line)
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"],padding="post",maxlen=250)
        predict = model.predict(encode)
        
        score = float(predict[0])
        result =""
        if float(predict[0]) > 0.85:
            result = "Very Good"
        elif float(predict[0]) > 0.65:
            result = "Good"
        elif(float(predict[0]) > 0.45):
            result = "Average"
        else:
            result = "bad"

    return render_template('sentiments.html', received_text=nline,score = score, verdict=result)




if __name__ == "__main__":
    init()
    app.run()