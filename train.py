import os
import pickle
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from keras._tf_keras.keras.layers import Input, Dense, Conv2D, Lambda, TimeDistributed, LSTM, GlobalAveragePooling2D, Flatten, Activation, MultiHeadAttention
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from keras_nac.layers import NAC, LAN
from ncps.wirings import AutoNCP
from ncps.keras import LTC, CfC

from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, prepare_data, create_sequences, load_pickle


DATA_DIR = "data"
DATA = os.path.join(DATA_DIR, "part1.pkl")

MODEL_NAME = "deepbot"
WEIGHTS_DIR = "model_weights"

TRAIN_SIZE = 0.8
VALID_SIZE = 0.2

SEQUENCE_LENGTH = 8
BATCH_SIZE = 64

EPOCHS = 200
LEARNING_RATE = 8.5e-05
SEED = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(WEIGHTS_DIR, exist_ok=True)

images, labels = load_pickle(DATA)
labels = np.asarray(labels, dtype=np.float32)

scaler = MinMaxScaler(feature_range=(-1, 1))
labels = scaler.fit_transform(labels)
with open(os.path.join(WEIGHTS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\nCombined dataset")
print("Images:", len(images))
print("Labels:", len(labels))

# Chronological train/validation/test split
num_samples = len(images)
train_end = int(num_samples * TRAIN_SIZE)

train_images = images[:train_end]
train_labels = labels[:train_end]

valid_images = images[train_end:]
valid_labels = labels[train_end:]


print("\nPreprocessing training images...")
train_images, train_labels = prepare_data(train_images, train_labels, is_training=True)

print("Preprocessing validation images...")
valid_images, valid_labels = prepare_data(valid_images, valid_labels, is_training=False)


print("\nProcessed images")
print("Train:", train_images.shape)
print("Valid:", valid_images.shape)


X_train, y_train = create_sequences(train_images, train_labels, SEQUENCE_LENGTH)
X_valid, y_valid = create_sequences(valid_images, valid_labels, SEQUENCE_LENGTH)



def build_model(rnn: str = "LAN"):
    inp=Input(shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS))
    x = TimeDistributed(Lambda(lambda x:x/127.5-1.0))(inp)
    x = TimeDistributed(Conv2D(64,(7,7),strides=(2,2),activation="relu",padding="same",kernel_regularizer=l2(1e-5)))(x)
    x = TimeDistributed(Conv2D(32,(7,7),strides=(2,2),activation="relu",padding="same",kernel_regularizer=l2(1e-5)))(x)
    x = TimeDistributed(Conv2D(64,(3,3),strides=(3,3),activation="relu",padding="same",kernel_regularizer=l2(3.02e-5)))(x)
    x = TimeDistributed(Conv2D(48,(7,7),strides=(3,3),activation="relu",padding="same",kernel_regularizer=l2(1e-5)))(x)
    x = TimeDistributed(Conv2D(24,(7,7),strides=(2,2),activation="relu",padding="same",kernel_regularizer=l2(1.4e-5)))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    if rnn=="NAC":
        x=NAC(d_model=64,num_heads=16,sparsity=0.8,topk=16,activation="tanh",return_sequences=False,use_riemann_sum=False)(x)

    elif rnn=="LSTM":
        x=LSTM(64,activation="tanh",return_sequences=False)(x)

    elif rnn=="LAN":
        x=LAN(d_model=64,num_heads=16,topk=16,activation="tanh",return_sequences=False)(x)

    elif rnn=="SDPA":
        x = MultiHeadAttention(key_dim=64, num_heads=16)(x, x)
        x =  Activation('tanh')(x)
        x = Flatten()(x)

    elif rnn=="LTC":
        x = Dense(32, activation="relu")(x) 
        out = LTC(AutoNCP(units=32,output_size=2), return_sequences=False)(x)
        model=Model(inputs=inp,outputs=out)
        return model
    
    elif rnn=="CfC":
        x = Dense(32, activation="relu")(x) 
        out = CfC(AutoNCP(units=32,output_size=2), return_sequences=False)(x)
        model=Model(inputs=inp,outputs=out)
        return model

TRAIN_POLICY = "LAN"
model = build_model(TRAIN_POLICY)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

model.summary()


best_weights_path = os.path.join(WEIGHTS_DIR, MODEL_NAME + "_" + TRAIN_POLICY + "_best.weights.h5")
final_model_path = os.path.join(WEIGHTS_DIR, MODEL_NAME + "_" + TRAIN_POLICY + ".keras")

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=1, min_lr=1e-10, verbose=0)
checkpoint = ModelCheckpoint(best_weights_path, monitor="val_loss", save_best_only=True, mode="min",verbose=0)
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)


history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_valid, y_valid),
    callbacks=[reduce_lr, checkpoint, early_stopping],
    verbose=1
)
model.save(final_model_path)

results = model.evaluate(
    X_valid,
    y_valid,
    batch_size=BATCH_SIZE,
    verbose=1
)
print(f"Test loss: {results[0]:.6f}")
print(f"Test MAE : {results[1]:.6f}")