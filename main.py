#https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
#http://nzw0301.github.io/2016/06/SCRN
#http://www.lai18.com/content/2086860.html?from=cancel
#

import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, activations, Merge, Activation, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
import h5py


class Data():

    #data_path = '/Users/user/6thsense/data/corpus_10000.txt'
    data_path = '/Volumes/USBHDD/corpus/keyakiblog/blog.txt'
    char_image_dic = {}
    image_size = 25
    timelength = 50

    def load(self):
        #output: ['こんにちは', '私は今日どこどこにいきます。', ... ]
        with open(self.data_path, 'r') as f:
            corpus = f.read().split('\n')
        return " ".join(corpus)
    
    def char2image(self, char):
        #input 私
        #output [私のImagevec]
        img1 = Image.new("L", (self.image_size, self.image_size), 0x00)
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype("/Library/Fonts/Osaka.ttf", 20, encoding="unic")
        draw.text(xy=(2,0) ,text= char, fill=(0xff), font=font)
        charvec = np.array(img1, dtype=np.int8).reshape((self.image_size * self.image_size))
        self.char_image_dic[char] = charvec
        return charvec

    def makeData(self, corpus):
        X = np.zeros((int(len(corpus)/self.timelength), self.timelength, self.image_size * self.image_size), dtype=np.int8)
        y = np.zeros((int(len(corpus)/self.timelength), self.timelength, self.image_size * self.image_size), dtype=np.int8)
        input_charvec = np.zeros((self.image_size * self.image_size))
        for i in range(int(len(corpus)/self.timelength)):
            print("{} / {}".format(i, int(len(corpus)/self.timelength)))
            for t in range(self.timelength):
                X[i][t][:] = input_charvec
                #探すのに時間がかかるのでは？
                if corpus[i] in self.char_image_dic.keys():
                    target_charvec = self.char_image_dic[corpus[i]]
                else:
                    target_charvec = self.char2image(corpus[i])
                y[i][t][:] = target_charvec
                input_charvec = target_charvec
        X_train = X[:int(len(X)*0.8)]
        y_train = y[:int(len(y)*0.8)]
        X_test = X[int(len(X)*0.2):]
        y_test = y[int(len(y)*0.2):]
        return X_train, y_train, X_test, y_test

    def saveData(self, X_train, y_train):
        with open('./char_image_dic', 'wb') as f:
            pickle.dump(self.char_image_dic, f)
        #データが大きすぎてpickelで処理できない。
        #with open('./X_train', 'wb') as f:
        #    pickle.dump(X_train, f)
        #with open('./y_train', 'wb') as f:
        #    pickle.dump(y_train, f)
            

class GenerateModel():
    
    text_image_dic = {}
    image_size = 25
    batch_size = 128
    timesteps = 50
    layer_num = 3
    Hidden_dim = 1000
    Output_dim = image_size * image_size
    
    def __init__(self):
        with open('./text_image_dic', 'rb') as f:
            self.text_image_dic = pickle.load(f)

    def buildModel(self, X_train, y_train):
        model = Sequential()
        model.add(LSTM(units=self.Hidden_dim, input_shape=(None, self.image_size*self.image_size), return_sequences=True))
        for i in range(self.layer_num - 1):
            model.add(LSTM(self.Hidden_dim, return_sequences=True))
        model.add(TimeDistributed(Dense(units=self.Output_dim,  activation='relu')))
        #SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.)
        model.compile(loss="cosine_proximity", optimizer="rmsprop")
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=5, validation_split=0.05, verbose=1)
        print(self.generateText(model, 30))
        return model
    
    def generateText(self, model, length):
        text = ''
        X = np.zeros((1, length, self.image_size * self.image_size))
        X[:, 0, :] = self.char2img('こ')
        for i in range(length):
            predictedImg = model.predict(X[:, :i+1, :])[0][-1]
            print(predictedImg)
            img = Image.fromarray(predictedImg.reshape((self.image_size, self.image_size)).astype(np.int8))
            img.save('./predictedImg/{}.png'.format(i))
            X[:, i, :] = predictedImg
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
        score = sorted(score.items(), key=lambda x:x[1], reverse=True)
        return score[0][0]

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            

if __name__ == '__main__':
    data = Data()
    corpus = data.load()
    X_train, y_train, X_test, y_test = data.makeData(corpus)
    data.saveData(X_train, y_train)
    print(X_train.shape)
    #データが大きすぎてpickelで処理できない。
    #with open('./X_train', 'rb') as f:
    #    X_train = pickle.load(f)
    #with open('./y_train', 'rb') as f:
    #    y_train = pickle.load(f)
    print("Data Loaded!!")
    generatemodel = GenerateModel()
    model = generatemodel.buildModel(X_train, y_train)
    print("Model Builded!!")
    model.save('my_model_back.h5')  # creates a HDF5 file 'my_model.h5'
    model = load_model('my_model_back_1.h5')
    for _ in range(5):
        print(generatemodel.generateText(model, length=10))