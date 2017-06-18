#https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
#http://nzw0301.github.io/2016/06/SCRN
#http://www.lai18.com/content/2086860.html?from=cancel
#

import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, activations, Merge, Activation, LSTM
from keras.optimizers import SGD
from keras.models import load_model
import h5py
from keras.layers.wrappers import TimeDistributed


class Data():

    data_path = '/Users/user/6thsense/data/corpus_10000.txt'
    char_image_dic = {}
    image_size = 25
    timelength = 30

    def __init__(self):
        pass

    def load(self):
        #output: ['こんにちは', '私は今日どこどこにいきます。', ... ]
        with open(self.data_path, 'r') as f:
            corpus = f.read().split('\n')
        return corpus
    
    def char2image(self, char):
        #input 私
        #output [私のImagevec]
        img1 = Image.new("L", (self.image_size, self.image_size), 0x00)
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype("/Library/Fonts/Osaka.ttf", 20, encoding="unic")
        draw.text(xy=(2,0) ,text= char, fill=(0xff), font=font)
        charvec = np.array(img1).reshape((self.image_size * self.image_size))
        self.char_image_dic[char] = charvec
        return charvec

    def makeData(self, corpus):
        X = np.zeros((0, self.image_size * self.image_size))
        y = np.zeros((len(corpus), self.timelength, self.image_size*self.image_size))
        input_charvec = np.zeros((self.image_size * self.image_size))
        #1文区切りずつ
        for i in range(len(corpus)):
            print('{} / {}'.format(i, len(corpus)))
            #1単語区切りずつ
            for j in range(len(corpus[i])):
                X = np.append(input_charvec, X)
                if corpus[i][j] in self.char_image_dic:
                    target_charvec = self.char_image_dic[corpus[i][j]]
                else:
                    target_charvec = self.char2image(corpus[i][j])
                y = np.append(target_charvec, y)
                input_charvec = target_charvec
        X = X.reshape((len(X)/self.timelength, self.timelength, self.image_size*self.image_size))
        y = y.reshape((len(y)/self.timelength, self.timelength, self.image_size*self.image_size))
        X_train = X[:int(len(X)*0.8)]
        y_train = y[:int(len(y)*0.8)]
        X_test = X[int(len(X)*0.2):]
        y_test = y[int(len(y)*0.2):]
        return X_train, y_train, X_test, y_test

    def saveData(self, X_train, y_train):
        with open('./text_image_dic', 'wb') as f:
            pickle.dump(self.char_image_dic, f)
        with open('./X_train', 'wb') as f:
            pickle.dump(X_train, f)
        with open('./y_train', 'wb') as f:
            pickle.dump(y_train, f)
            

class GenerateModel():
    
    text_image_dic = {}
    image_size = 25
    batch_size = 100
    timesteps = 30
    layer_num = 2
    Hidden_dim = 300
    Output_dim = image_size * image_size
    
    def __init__(self):
        with open('./text_image_dic', 'rb') as f:
            self.text_image_dic = pickle.load(f)

    def buildModel(self, X_train, y_train):
        model = Sequential()
        model.add(LSTM(units=self.Hidden_dim, input_shape=(None, self.V), return_sequences=True))
        for i in range(self.layer_num - 1):
            model.add(LSTM(self.Hidden_dim, return_sequences=True))
        model.add(TimeDistributed(Dense(units=self.Output_dim,  activation='relu')))
        SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss="mean_squared_error", optimizer="sgd")
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=1, validation_split=0.05, verbose=1)
        return model
    
    def generateText(self, model, length):
        text = ''
        X = np.zeros((1, length, self.image_size * self.image_size))
        for i in range(length):
            predictedImg = model.predict(X[:, i+1, :])[0]
            img = Image.fromarray(predictedImg.reshape((self.image_size, self.image_size)))
            img.save('./predictedImg/{}.png'.format(i))
            X[:, i] = predictedImg
            predictedChar = self.img2char(predictedImg)
            text += predictedChar
        return text

    def char2img(self, char):
        img1 = Image.new("L", (self.image_size, self.image_size), 0x00)
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype("/Library/Fonts/Osaka.ttf", 20, encoding="unic")
        draw.text(xy=(2,0) ,text=char, fill=(0xff), font=font)
        char_array = np.array(img1).reshape(self.image_size*self.image_size)
        return char_array

    def img2char(self, predictedImg):
        score = {}
        for char, charVec in self.text_image_dic.items():
            score[char] = self.cos_sim(predictedImg, charVec)
        score = sorted(score.items(), key=lambda x:x[1])
        return score[0]

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            

if __name__ == '__main__':
    data = Data()
    corpus = data.load()
    X_train, y_train, X_test, y_test = data.makeData(corpus)
    data.saveData(X_train, y_train)
    print(X_train.shape)
    with open('./X_train', 'rb') as f:
        X_train = pickle.load(f)
    with open('./y_train', 'rb') as f:
        y_train = pickle.load(f)
    print("Data Loaded!!")
    generatemodel = GenerateModel()
    model = generatemodel.buildModel(X_train, y_train)
    print("Model Builded!!")
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    model = load_model('my_model.h5')
    print(generatemodel.generateText(model, length=10))