#MAX_EPOCHS = 20

import keras.models
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM

def compile_and_fit(model,trainX,trainY,testX,testY,BATCH_SIZE, MAX_EPOCHS,LR):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,mode='min')
  model.compile(loss=tf.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(learning_rate=LR),metrics=[tf.metrics.MeanAbsoluteError()])  #0.001 0.0001 0.01
  history = model.fit( x=trainX,   y=trainY,
    validation_data=(testX, testY),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE )
                    
    #validation_data=validation_data,callbacks=[early_stopping])
  return history

def red_neuronal1():
    inputs  = Input(shape=(6,1))
    x = LSTM(64,activation="relu",return_sequences=True)(inputs)
    x = Dropout(0.7)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.7)(x)
    x = Dense(16, activation="relu")(x)
    lstm_model = Model(inputs, x)
    return lstm_model