from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D
from keras import optimizers
from keras import losses
from keras import activations
from keras import initializers


def CNN(N_output=10, kernel_size=(3,3), stride=1, pool_size=(2,2), data_shape=(32,32,1), learning_rate=1E-4, decay_rate=1E-4): 
    act = activations.relu
    padding = 'same'

    input_img= Input(shape=data_shape)
    layer_1 = Conv2D(filters=64, kernel_size=(5,5), activation=act, strides=stride, padding=padding)(input_img)
    layer_1 = MaxPooling2D(pool_size=(2,2), padding=padding)(layer_1)

    layer_2 = Conv2D(filters=32, kernel_size=(9,9), activation=act, strides=stride, padding=padding)(layer_1)
    layer_2 = MaxPooling2D(pool_size=(4,4), padding=padding)(layer_2)

    layer_3 = Conv2D(filters=8, kernel_size=(3,3), activation=act, strides=stride, padding=padding)(layer_2)
    layer_3 = MaxPooling2D(pool_size=(2,2), padding=padding)(layer_3)

    layer_f = Flatten()(layer_3)
    layer_d = Dense(units=N_output, activation=act)(layer_f)

    model = Model(input_img, layer_d)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

    model.compile(loss=losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
    from keras import backend as K
    print('GPU:')
    print(K.tensorflow_backend._get_available_gpus())
    return model
