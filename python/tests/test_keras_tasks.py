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
    from keras.layers import Dense, Dropout, Activation
    mlp = Sequential()
    mlp.add(Dense(64, input_dim=20, init='uniform'))
    mlp.add(Activation('tanh'))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(64, init='uniform'))
    mlp.add(Activation('tanh'))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(10, init='uniform'))
    mlp.add(Activation('softmax'))
    mlp.compile(loss='categorical_crossentropy', \
                  optimizer='sgd',
                  metrics=['accuracy'])
    return mlp
def create_CNN(path_to_serialized_model=""):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    cnn = Sequential()
    cnn.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(32, 3, 3))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Convolution2D(64, 3, 3, border_mode='valid'))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(64, 3, 3))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())
    # Note: Keras does automatic shape inference.
    cnn.add(Dense(256))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.5))

    cnn.add(Dense(10))
    cnn.add(Activation('softmax')) 
    cnn.compile(loss='categorical_crossentropy', optimizer='sgd')
    return cnn

def create_RNN(path_to_serialized_model=""):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import Embedding
    from keras.layers import LSTM
    max_features = 300
    maxlen=20
    rnn = Sequential()
    rnn.add(Embedding(max_features, 256, input_length=maxlen))
    rnn.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    rnn.add(Dropout(0.5))
    rnn.add(Dense(1))
    rnn.add(Activation('sigmoid'))

    rnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return rnn

def create_CNN_RNN_merge(path_to_serialized_model=""):
    from keras.models import Sequential
    from keras.layers.wrappers import TimeDistributed

    cnnrnn_merge = Sequential()
    
    max_caption_len = 16
    vocab_size = 10000

    # first, let's define an image model that
    # will encode pictures into 128-dimensional vectors.
    # it should be initialized with pre-trained weights.
    cnn = Sequential()
    cnn.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(32, 3, 3))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Convolution2D(64, 3, 3, border_mode='valid'))
    cnn.add(Activation('relu'))
    cnn.add(Convolution2D(64, 3, 3))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(128))

    # let's load the weights from a save file.
    cnn.load_weights('weight_file.h5')

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    rnn = Sequential()
    rnn.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    rnn.add(GRU(output_dim=128, return_sequences=True))
    rnn.add(TimeDistributed(Dense(128)))

    # let's repeat the image vector to turn it into a sequence.
    cnn.add(RepeatVector(max_caption_len))

    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    cnnrnn_merge = Sequential()
    cnnrnn_merge.add(Merge([cnn, rnn], mode='concat', concat_axis=-1))
    # let's encode this vector sequence into a single vector
    cnnrnn_merge.add(GRU(256, return_sequences=False))
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    cnnrnn_merge.add(Dense(vocab_size))
    cnnrnn_merge.add(Activation('softmax'))

    cnnrnn_merge.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    cnnrnn_merge
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
#Testing creation of cArl ANN objects                                            #
##################################################################################

def test_create_cArl_ANN_from_keras_MLP_object():
    mlp = create_MLP()
    ann = ArtificialNeuralNetwork(mlp, keras)
    #TODO Run tests to see if the model was created successfully

def test_create_cArl_ANN_from_keras_CNN_object():
    cnn = create_CNN()
    ann = ArtificialNeuralNetwork(cnn, keras)
    #TODO Run tests to see if the model was created successfully

def test_create_cArl_ANN_from_keras_RNN_object():
    rnn = create_RNN()
    ann = ArtificialNeuralNetwork(rnn, keras)
    #TODO Run tests to see if the model was created successfully

def test_create_cArl_ANN_from_keras_CNN_RNN_merge_object():
    cnnrnn_merge = create_CNN_RNN_merge()
    ann = ArtificialNeuralNetwork(ann, keras)
    #TODO Run tests to see if the model was created successfully
    

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



