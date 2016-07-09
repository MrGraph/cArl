from __future__ import print_function
import logging


from ..src.RecommendationObject import RecommendationObject
from ..src.RecommendationObject import 

TEST_SUCCESS = True
TEST_FAILURE = False
keras = "Keras"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keras_imports_and_version():
    try:
        import keras
        from keras.models import Sequential
    except:
        #Log it
        logger.error('Keras does not appear to be properly installed.')

    assert(int(keras.__version__.replace('.', '')) >= 104), logger.error('The Keras version is too low. Please upgrade to at least 1.0.4')

#Recommend weights from in memory model objects
#Input is a boolean indicating whether to test writing to a file,
#or simply return a recommendation object
def test_keras_weight_recommendation_from_in_memory_MLP(keras_model):
    #Create cArl ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(keras_model, keras)
    recommendation = recommend_weight_initialization(ann)
    assert(recommendation != None), logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_MLP is NoneType. Recommendation failed')
    assert("weight_initialization_recommendation" in recommendation.provenance_dict, logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_MLP does not contain a weight initialization recommendation. Recommendation failed') 
    

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
