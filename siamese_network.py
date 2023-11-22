from keras.datasets import cifar10
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Reshape, GlobalAveragePooling2D, Dense
from keras.layers import Concatenate
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
import keras.optimizers as opt
import numpy as np
import itertools
import matplotlib.pyplot as plt

def get_sample(X, y, range):
    rand_idxs = np.random.choice(X.shape[0], range, replace=False)
    return X[rand_idxs], y[rand_idxs]
#for 300 itens there are 300**2 pairs that serve as input

def make_paired_dataset(X,y):
    X_pairs, y_pairs = [] , []

    tuples = [(x1,y1) for x1,y1 in zip(X,y)]
    for t in itertools.product(tuples, tuples):
        # pairA, pairB = t
        imgA, labelA = t[0]
        imgB, labelB = t[1]

        new_label = int(labelA == labelB)
        X_pairs.append([imgA, imgB])
        y_pairs.append(new_label)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    return X_pairs, y_pairs

def get_cnn_block(depth = 64):
    model = Sequential()
    model.add(Conv2D(depth, 3, 1))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train_sample, y_train_sample = get_sample(X_train, y_train, 250)
X_train_pairs, y_train_pairs = make_paired_dataset(X_train_sample, y_train_sample)

X_test_sample, y_test_sample = get_sample(X_test, y_test, 125)
X_test_pairs, y_test_pairs = make_paired_dataset(X_test_sample, y_test_sample)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# print(X_train[0])

shape = (32,32,3)
inputA = Input(shape,name='inp_A')
inputB = Input(shape,name='inp_B')

# from the 784 pixes return a 64 feaature vector
DEPTH = 64
cnn = Sequential([get_cnn_block(DEPTH),
                  get_cnn_block(DEPTH*2),
                  get_cnn_block(DEPTH*4),
                  get_cnn_block(DEPTH*8),
                  GlobalAveragePooling2D(),
                  Dense(32, activation='relu')])

feature_vector_A = cnn(inputA)
feature_vector_B = cnn(inputB)

concat = Concatenate()([feature_vector_A, feature_vector_B])
dense = Dense(32, activation='relu')(concat)
output = Dense(1,activation='sigmoid')(dense)

model = Model(inputs=[inputA, inputB], outputs=output)
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = opt.Adam(learning_rate=0.01), metrics=['accuracy'])

es = EarlyStopping(patience=3)

model.fit(x=[X_train_pairs[:, 0, :, :], X_train_pairs[:,1,:,:]], y =y_train_pairs,  
          validation_data=([X_test_pairs[:, 0, :, :], X_test_pairs[:,1,:,:]], y_test_pairs),
          epochs = 200,
          batch_size=32,
          callbacks=[es])

imgA, imgB = X_test[0], X_test[8]
labelA, labelB = y_test[0], y_test[8]

print(labelA, labelB)

plt.figure(dpi=28)
plt.imshow(imgA)

plt.figure(dpi=28)
plt.imshow(imgB)

prediction = model.predict([imgA, imgB]).flatten()[0] > 0.5
print(prediction)