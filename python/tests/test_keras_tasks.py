from __future__ import print_function
import logging

TEST_SUCCESS = True
TEST_FAILURE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keras_imports_and_version():
    try:
        import keras
        from keras.models import Sequential
    except:
        #Log it
        logger.error('Keras does not appear to be properly installed.')

    if int(keras.__version__.replace('.', '')) < 104:
        logger.error('The Keras version is too low. Please upgrade to at least 1.0.4')

#Recommend weights from in memory model objects
#Input is a boolean indicating whether to test writing to a file,
#or simply return a recommendation object
def test_keras_weight_recommendation_from_in_memory_MLP():
def test_keras_weight_recommendation_from_in_memory_CNN():
def test_keras_weight_recommendation_from_in_memory_RNN():
def test_keras_weight_recommendation_from_in_memory_CNN_RNN_merge():

#Recommend weights from serialized model objects
#Input is a boolean indicating whether to test writing to a file,
#or simply return a recommendation object
def test_keras_weight_recommendation_from_serialized_MLP():
def test_keras_weight_recommendation_from_serialized_CNN():
def test_keras_weight_recommendation_from_serialized_RNN():
def test_keras_weight_recommendation_from_serialized_CNN_RNN_merge():
