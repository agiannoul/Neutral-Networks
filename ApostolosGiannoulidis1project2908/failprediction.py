import keras
from keras.datasets import mnist
from keras.models import Sequential


# Prepare the data
(x_train1, y_train), (x_test1, y_test) = mnist.load_data()
#transform the shape in :  (number of shample) , (28X28 = 784) 
#scale the data to [0,1]
x_train = x_train1.reshape((-1, 28*28))/255
x_test = x_test1.reshape((-1, 28*28))/255
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = keras.models.load_model('my_modeel_1')

pos=-1
for i in range(0,10000):
    k= np.array(x_test[i])
    k=k.reshape(1,784)
    pr=model.predict(k)
    if np.argmax(pr)!=np.argmax(y_test[i]):
        pos=i
        break
if pos != -1 :
    k= np.array(x_test[pos])
    k=k.reshape(1,784)
    pr=model.predict(k)
    print(" Prediction : ",end=" ")
    print(np.argmax(pr))
    print(" Real value : ",end=" ")
    print(np.argmax(y_test[pos]))

    import matplotlib.pyplot as plt

    plt.imshow(x_test1[pos], cmap=plt.get_cmap('gray'))
    plt.show()
