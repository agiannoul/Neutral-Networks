import keras
from keras.datasets import mnist
from keras.models import Sequential

# Prepare the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#transform the shape in :  (number of shample) , (28X28 = 784) 
#scale the data to [0,1]
x_train = x_train.reshape((-1, 28*28))/255
x_test = x_test.reshape((-1, 28*28))/255
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# we create a sequential model with keras , 
# in that way we built a model that represents a neural network layer bu layer
model = Sequential()
#hideen layer fully conected with 32 neurals ,activation function relu
model.add(Dense(256,use_bias=True,activation='relu',input_dim=784))
#Output layer fully conected with 10 neurals,activation function softmax ( represent the data as 
# propabilities with softmax function , all outputs values sums in 1 )
model.add(Dense(10, activation='softmax'))
import time

start = time.clock()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.95, nesterov=False),
              metrics=['accuracy'])

results = model.fit(x_train, y_train,
          batch_size=256,
          epochs=50,
          verbose=2,
          validation_data=(x_test, y_test))


print("Time : ",time.clock() - start)

model.save('my_modeel_1')