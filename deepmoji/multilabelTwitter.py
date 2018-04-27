"""Trains the DeepMoji architecture on the IMDB sentiment classification task.
   This is a simple example of using the architecture without the pretrained model.
   The architecture is designed for transfer learning - it should normally
   be used with the pretrained model for optimal performance.
   # Should be run from DeepMoji/deepmoji.
"""
from __future__ import print_function
import keras
from typing import Tuple, List, Dict
#import example_helper
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.datasets import imdb
from model_def import deepmoji_multilabel_architecture

# Preprocessing:
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Val loss:
from keras import backend as K

# Call backs:
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from GetBest import GetBest

# Evaluate:
from sklearn.metrics import jaccard_similarity_score

import warnings
import logging
import sys


# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#Set up file handler:
#fh = logging.FileHandler(os.path.basename(__file__) + str(datetime.now()).replace(" ", "_") + '.log')
# Pipe output to stdout
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


# Function definitions:
def jaccard_distance(y_true, y_pred, smooth=100):
    # References:
    # https://stackoverflow.com/questions/49284455/keras-custom-function-implementing-jaccard
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_score_K(y_true, y_pred):
    # References:
    # https://stackoverflow.com/questions/49284455/keras-custom-function-implementing-jaccard
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    return (intersection) / (sum_ - intersection)
    

def maxLikelihoodToBinary(A: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (A > thresh).astype(np.int)

def convertToDF(A: np.ndarray, colnames: List[str]) -> pd.DataFrame:
    """
    Converts numpy array to pandas dataframe with column names
    """
    return pd.DataFrame(A, columns=colnames)

def saveSubmission(predictions: pd.DataFrame, write_path:str):
	# TODO: add read path and write path:
    submission = pd.read_csv('/Users/seanmhendryx/NeuralNets/graduate-student-project-SMHendryx/data/2018-E-c-En-dev.txt', sep = '\t', usecols =['ID', 'Tweet'])
    #df_c = pd.concat([df_a.reset_index(), df_b], axis=1)
    submission = pd.concat([submission, predictions], axis = 1)
    submission.to_csv(write_path, sep = '\t', index = False)
    logger.info('Wrote to %s' % write_path)

def loadTrain(max_features: int, maxlen = None) -> Tuple[np.ndarray, np.ndarray]:
	# TODO: add path argument
    df = pd.read_csv('/Users/seanmhendryx/NeuralNets/graduate-student-project-SMHendryx/data/2018-E-c-En-train.txt', sep = '\t')
    features = df[['Tweet']]
    features = prepFeatures(features, max_features, maxlen)
    columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy','love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    labels = df[columns]
    return features, labels.values

def loadDev(max_features: int, maxlen = None) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: add path argument
    df = pd.read_csv('/Users/seanmhendryx/NeuralNets/graduate-student-project-SMHendryx/data/2018-E-c-En-dev.txt', sep = '\t')
    features = df[['Tweet']]
    features = prepFeatures(features, max_features, maxlen)
    columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy','love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    labels = df[columns]
    return features, labels.values

def prepFeatures(df: pd.DataFrame, max_vocab_size: int = 20000, maxlen= 300, colname: str = 'Tweet') -> np.ndarray:
    """
    Prep features for NLP from strings
    # reference: http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
    """
    tokenizer = Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(df[colname])
    sequences = tokenizer.texts_to_sequences(df[colname])
    logger.info('Pad sequences (samples x time)')
    return pad_sequences(sequences, maxlen=maxlen)

def saveModel(model: keras.Model, path: str):
    # path where to save hdf5 model trained weights
    model.save(path)

def rbind(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # row binds two pandas DataFrames:
    return pd.concat([df1, df2])

#def main():
num_classes = 11

# Seed for reproducibility
np.random.seed(1337)

nb_tokens = 20000
maxlen = 80
batch_size = 32

logger.info('Loading data...')
# I am here:
X_train, y_train = loadTrain(nb_tokens, maxlen)
colnames = list(y_train.columns)

X_test, y_test = loadDev(nb_tokens, maxlen)





#(iX_train, iy_train), (iX_test, iy_test) = imdb.load_data(num_words=nb_tokens)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = deepmoji_multilabel_architecture(nb_classes=num_classes, nb_tokens=nb_tokens, maxlen=maxlen, final_dropout_rate = 0.5)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy', jaccard_score_K])


# Define early stopping with callbacks:
patience = 10
early_stopping = EarlyStopping(monitor='val_jaccard_score_K', min_delta=0, patience=patience, verbose=1, mode='max')
callbacks = [early_stopping, GetBest(monitor='val_jaccard_score_K', verbose=1, mode='max')]

# Plot learning with tensorboard:
tensorboard=True
if tensorboard:
    tb = TensorBoard(log_dir='../tensorboards/', histogram_freq=1, write_graph=True, write_images=False)
    callbacks.append(tb)    


logger.info('Training...')
model.fit(X_train, y_train, batch_size=batch_size, epochs=100, validation_data=(X_test, y_test), callbacks = callbacks)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
logger.info('Dev vanilla score: %s' % score)
logger.info('Dev vanilla accuracy: %s' % acc)

# Evaluate
probabilities = model.predict(X_test)
y_hat = maxLikelihoodToBinary(probabilities)
# sklearn.metrics.jaccard_similarity_score(y_true, y_pred, normalize=True, sample_weight=None)
score = jaccard_similarity_score(y_test, y_hat, normalize=True, sample_weight=None)
logger.info('sklearn jaccard_similarity_score: %s' % score)
# sklearn jaccard_similarity_score: 0.2169353971837042

# merge train and dev:
#X = rbind(X_test, )

#Save model:
num_epochs = GetBest.get_best_num_epochs()
path_to_save_weights = '/Users/seanmhendryx/NeuralNets/graduate-student-project-SMHendryx/weights/E-C_en_pred_' + model.name + '_' + num_epochs + '_epochs.hdf5'
model.save(path_to_save_weights)
# to load: from keras.models import load_model; model = load_model(path_to_save_weights)

write_path = '/Users/seanmhendryx/NeuralNets/graduate-student-project-SMHendryx/data/E-C_en_pred.txt'
saveSubmission(convertToDF(y_hat, colnames), write_path)



