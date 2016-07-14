from __future__ import print_function
import logging


from ..src.RecommendationObject import RecommendationObject
from ..src.RecommendationObject import 

TEST_SUCCESS = True
TEST_FAILURE = False
keras = "Keras"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################################################################################
#Functions for creating various Keras models for testing                         #
##################################################################################
def create_MLP(path_to_serialized_model=""):
    from keras.models import Sequential
    mlp = Sequential()
    return mlp
def create_CNN(path_to_serialized_model=""):
    from keras.models import Sequential
    cnn = Sequential()
    return cnn
def create_RNN(path_to_serialized_model=""):
    from keras.models import Sequential
    rnn = Sequential()
    return rnn
def create_CNN_RNN_merge(path_to_serialized_model=""):
    from keras.models import Sequential
    cnnrnn_merge = Sequential()
    return cnnrnn_merge

##################################################################################
#Testing necessary Keras imports                                                 #
##################################################################################

def test_keras_imports_and_version():
    try:
        import keras
        from keras.models import Sequential, model_from_json
        from keras.layers import Dense
    except:
        #Log it
        logger.error('Keras does not appear to be properly installed.')

    assert(int(keras.__version__.replace('.', '')) >= 104), logger.error('The Keras version is too low. Please upgrade to at least 1.0.4')


##################################################################################
#Testing necessary Keras imports                                                 #
##################################################################################

def test_create_cArl_ANN_from_keras_MLP_object():
    mlp_model = create_MLP()
    ann = ArtificialNeuralNetwork(mlp_model, keras)
    #TODO Run tests to see if the model was created successfully

##################################################################################
#Recommend weights from in memory model objects                                  #
##################################################################################

def test_keras_weight_recommendation_from_in_memory_MLP():
    mlp_model = create_MLP()
    #Create cArl ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(mlp_model, keras)
    recommendation = recommend_weight_initialization(ann)
    assert(recommendation != None), logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_MLP is NoneType. Recommendation failed')
    assert("weight_initialization_recommendation" in recommendation.provenance_dict, logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_MLP does not contain a weight initialization recommendation. Recommendation failed') 
    

def test_keras_weight_recommendation_from_in_memory_CNN():
    cnn_model = create_CNN()
    #Create cArl ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(cnn_model, keras)
    recommendation = recommend_weight_initialization(ann)
    assert(recommendation != None), logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_CNN is NoneType. Recommendation failed')
    assert("weight_initialization_recommendation" in recommendation.provenance_dict, logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_CNN does not contain a weight initialization recommendation. Recommendation failed') 
    

def test_keras_weight_recommendation_from_in_memory_RNN():
    rnn_model = create_RNN()
    #Create cArl ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(rnn_model, keras)
    recommendation = recommend_weight_initialization(ann)
    assert(recommendation != None), logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_RNN is NoneType. Recommendation failed')
    assert("weight_initialization_recommendation" in recommendation.provenance_dict, logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_RNN does not contain a weight initialization recommendation. Recommendation failed') 


def test_keras_weight_recommendation_from_in_memory_CNN_RNN_merge():
    cnn_rnn_merge = create_CNN_RNN_merge()
    ann = ArtificialNeuralNetwork(cnn_rnn_merge, keras)
    recommendation = recommend_weight_initialization(ann)
    assert(recommendation != None), logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_CNN_RNN_merge is NoneType. Recommendation failed')
    assert("weight_initialization_recommendation" in recommendation.provenance_dict, logger.error('Recommendation object created via test_keras_weight_recommendation_from_in_memory_CNN_RNN_merge does not contain a weight initialization recommendation. Recommendation failed') 


##################################################################################
#Test that ANNs read in from serializations are equivalent to those              #
#created by in memory Keras objects                                              #
##################################################################################
def test_create_cArl_ANN_from_keras_MLP_json():
    path_to_serialized_model = "./mlp.json"
    mlp = create_MLP(path_to_serialized_model)
    #TODO Is constructor overloading allowed in python?
    ANN_from_json_serialization = ArtificialNeuralNetwork(path_to_serialized_model, keras)
    ANN_from_in_memory_object = ArtificialNeuralNetwork(mlp, keras)
    #TODO Run tests to see if the models were created successfully
    #TODO Delete file

def test_create_cArl_ANN_from_keras_CNN_json():
    path_to_serialized_model = "./cnn.json"
    cnn = create_CNN(path_to_serialized_model)
    #TODO Is constructor overloading allowed in python?
    ANN_from_json_serialization = ArtificialNeuralNetwork(path_to_serialized_model, keras)
    ANN_from_in_memory_object = ArtificialNeuralNetwork(cnn, keras)
    #TODO Run tests to see if the models were created successfully
    #TODO Delete file

def test_create_cArl_ANN_from_keras_RNN_json():
    path_to_serialized_model = "./rnn.json"
    rnn = create_CNN(path_to_serialized_model)
    #TODO Is constructor overloading allowed in python?
    ANN_from_json_serialization = ArtificialNeuralNetwork(path_to_serialized_model, keras)
    ANN_from_in_memory_object = ArtificialNeuralNetwork(rnn, keras)
    #TODO Run tests to see if the models were created successfully
    #TODO Delete file


def test_create_cArl_ANN_from_keras_CNN_RNN_merge_json():
    path_to_serialized_model = "./cnn_rnn_merged.json"
    cnn_rnn_merged = create_CNN(path_to_serialized_model)
    #TODO Is constructor overloading allowed in python?
    ANN_from_json_serialization = ArtificialNeuralNetwork(path_to_serialized_model, keras)
    ANN_from_in_memory_object = ArtificialNeuralNetwork(cnn_rnn_merged, keras)
    #TODO Run tests to see if the models were created successfully
    #TODO Delete file



