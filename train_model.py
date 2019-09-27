import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_label),(test_data, test_label) = data.load_data(num_words=10000)



word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#model down here

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

#print(model.summary())

model.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_label[:10000]
y_train = train_label[10000:]

fitModel = model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)

model.save('classify-1.h5')

results = model.evaluate(test_data,test_label)

#print(results)



model  = keras.models.load_model('classify-1.h5')

def review_encode(s):
    encode = [1]

    for word in s:
        if word in word_index:
            encode.append(word_index[word.lower()])
        else:
            encode.append(2)

    return encode
text ='I watched the film today and I have come to the conclusion that this Disney film is one classic that will keep on delivering to Disney fans like myself...TLK is a wonderful film with great characters and songs with a powerful message that you can always find your way back no matter how hard the journey. I think The Lion King is one of the best Disney animated films matching with the power of Bambi and Brother Bear. I love how Disney Works hard to deliver the best in Animation,Song,And Magic'

nline = text.replace(",","").replace(".","").replace(")","").replace("(","").replace(":","").replace("\\","").strip()
encode = review_encode(nline)
encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"],padding="post",maxlen=250)
predict = model.predict(encode)
print(nline)
print(encode)
print(predict[0])
