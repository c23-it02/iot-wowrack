import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Lambda, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def buat_siamese(input_shape, embedding_dim=128):
    inp = Input(input_shape)
    
    #first convo
    c = Conv2D(4, (3,3), padding='same')(inp)
    n = BatchNormalization()(c)
    a = tf.keras.layers.Activation('relu')(n)
    m = MaxPooling2D((2,2), padding='same')(a)
    
    #second convo
    c = Conv2D(8, (3,3), padding='same')(m)
    n = BatchNormalization()(c)
    a = tf.keras.layers.Activation('relu')(n)
    m = MaxPooling2D((2,2), padding='same')(a)
    
    #third convo
    c = Conv2D(16, (3,3), padding='same')(m)
    n = BatchNormalization()(c)
    a = tf.keras.layers.Activation('relu')(n)
    m = MaxPooling2D((2,2), padding='same')(a)
       
    #fourt convo
    c = Conv2D(32, (3,3), padding='same')(m)
    n = BatchNormalization()(c)
    a = tf.keras.layers.Activation('relu')(n)
    m = MaxPooling2D((2,2), padding='same')(a)
    
    #c = Conv2D(128, (3,3), activation='relu')(m)
    
    #fc
    f = Flatten()(m)
    d = Dense(1024, activation='relu')(f)
    drop = Dropout(0.2)(d)
    output = Dense(embedding_dim)(drop)
    return Model(inputs=inp , outputs=output)
