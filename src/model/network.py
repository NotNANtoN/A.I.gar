import heapq
import keras
import numpy
import tensorflow as tf
import importlib.util
import keras.backend as K
from keras.layers import Dense, LSTM, Softmax
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model


from .parameters import *

class Network(object):

    def __init__(self, trainMode, modelName, parameters):
        self.parameters = parameters
        self.trainMode = trainMode

        self.actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
                   eject in [0, 1]]
        # Filter out actions that do a split and eject at the same time
        # Also filter eject actions for now
        for action in self.actions[:]:
            if action[2] or action[3]:
                self.actions.remove(action)

        self.num_actions = len(self.actions)
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN


        self.gpus = self.parameters.GPUS

        # Q-learning
        self.targetNetworkSteps = self.parameters.TARGET_NETWORK_STEPS
        self.targetNetworkMaxSteps = self.parameters.TARGET_NETWORK_MAX_STEPS
        self.discount = self.parameters.DISCOUNT
        self.epsilon = self.parameters.EPSILON
        self.frameSkipRate = self.parameters.FRAME_SKIP_RATE
        self.gridSquaresPerFov = self.parameters.GRID_SQUARES_PER_FOV

        # ANN
        self.learningRate = self.parameters.ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncLSTM = self.parameters.ACTIVATION_FUNC_LSTM
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3

        if self.parameters.EXP_REPLAY_ENABLED:
            input_shape_lstm = (self.parameters.MEMORY_TRACE_LEN, self.stateReprLen)
            stateful_training = False
            self.batch_len = self.parameters.MEMORY_BATCH_LEN

        else:
            input_shape_lstm = (1, self.stateReprLen)
            stateful_training = True
            self.batch_len = 1

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + self.num_actions))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.valueNetwork = Sequential()
                self.valueNetwork.add(Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                                           bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.valueNetwork.add(Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.valueNetwork.add(Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                                               , kernel_initializer=initializer))
                self.valueNetwork.add(Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                                       , kernel_initializer=initializer))
                self.valueNetwork = multi_gpu_model(self.valueNetwork, gpus=self.gpus)
        else:
            self.valueNetwork = Sequential()
            hidden1 = None
            if self.parameters.NEURON_TYPE == "MLP":
                hidden1 = Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                                          bias_initializer=initializer, kernel_initializer=initializer)
            elif self.parameters.NEURON_TYPE == "LSTM":
                hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm,
                               return_sequences = True, stateful= stateful_training, batch_size=self.batch_len)

            self.valueNetwork.add(hidden1)
            #self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, return_sequences=True, stateful=stateful_training, batch_size=self.batch_len)
                self.valueNetwork.add(hidden2)
                #self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, return_sequences=True, stateful=stateful_training, batch_size=self.batch_len)
                self.valueNetwork.add(hidden3)
                #self.valueNetwork.add(Dropout(0.5))

            if self.parameters.NEURON_TYPE == "MLP": #or self.parameters.NEURON_TYPE == "LSTM":
                self.valueNetwork.add(
                    Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                          , kernel_initializer=initializer))
            elif self.parameters.NEURON_TYPE == "LSTM":
                self.valueNetwork.add(LSTM(self.num_actions, activation=self.activationFuncOutput,
                                   return_sequences=True, stateful=stateful_training, batch_size=self.batch_len))

        self.targetNetwork = keras.models.clone_model(self.valueNetwork)
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())



        if self.optimizer == "Adam":
            optimizer = keras.optimizers.Adam(lr=self.learningRate)
        else:
            optimizer = keras.optimizers.SGD(lr=self.learningRate)

        self.optimizer = optimizer

        self.valueNetwork.compile(loss='mse', optimizer=optimizer)
        self.targetNetwork.compile(loss='mse', optimizer=optimizer)

        if self.parameters.NEURON_TYPE == "LSTM":
            # We predict using only one state
            input_shape_lstm = (1, self.stateReprLen)
            self.actionNetwork = Sequential()
            hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm,
                            return_sequences=True, stateful=True, batch_size=1)
            self.actionNetwork.add(hidden1)

            if self.hiddenLayer2 > 0:
                hidden2 = LSTM(self.hiddenLayer2, return_sequences=True, stateful=True, batch_size=self.batch_len)
                self.actionNetwork.add(hidden2)
            if self.hiddenLayer3 > 0:
                hidden3 = LSTM(self.hiddenLayer3, return_sequences=True, stateful=True, batch_size=self.batch_len)
                self.actionNetwork.add(hidden3)
            self.actionNetwork.add(LSTM(self.num_actions, activation=self.activationFuncOutput,
                                        return_sequences=False, stateful=True, batch_size=self.batch_len))
            self.actionNetwork.compile(loss='mse', optimizer=optimizer)

        if modelName is not None:
            self.load(modelName)


    def reset_general(self, model):
        session = K.get_session()
        for layer in model.layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(layer.name, v))


    def reset_weights(self):
        self.reset_general(self.valueNetwork)
        self.reset_general(self.targetNetwork)

    def reset_hidden_states(self):
        self.actionNetwork.reset_states()
        self.valueNetwork.reset_states()
        self.targetNetwork.reset_states()

    def load(self, modelName):
        path = modelName
        self.loadedModelName = modelName
        self.valueNetwork = load_model(path + "model.h5")
        self.targetNetwork = load_model(path + "model.h5")

    def trainOnBatch(self, inputs, targets):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.valueNetwork.train_on_batch(inputs, targets)
            else:
                return self.valueNetwork.train_on_batch(numpy.array([numpy.array([inputs])]), numpy.array([numpy.array([targets])]))
        else:
            return self.valueNetwork.train_on_batch(inputs, targets)

    def updateActionNetwork(self):
        self.actionNetwork.set_weights(self.valueNetwork.get_weights())

    def updateTargetNetwork(self):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())

    def predict(self, state, batch_len = 1):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.valueNetwork.predict(state, batch_size=batch_len)
            else:
                return self.valueNetwork.predict(numpy.array([numpy.array([state])]))[0][0]
        return self.valueNetwork.predict(numpy.array([state]))[0]

    def predict_target_network(self, state, len_batch = 1):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.targetNetwork.predict(state, batch_size=len_batch)
            else:
                return self.targetNetwork.predict(numpy.array([numpy.array([state])]))[0][0]
        return self.targetNetwork.predict(numpy.array([state]))[0]

    def predict_action_network(self, trace):
        return self.actionNetwork.predict(numpy.array([numpy.array([trace])]))[0]

    def saveModel(self, path):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())
        self.targetNetwork.save(path + "model.h5")

    def setEpsilon(self, val):
        self.epsilon = val

    def setFrameSkipRate(self, value):
        self.frameSkipRate = value

    def getTrainMode(self):
        return self.trainMode

    def getMemoriesPerUpdate(self):
        return self.memoriesPerUpdate

    def getMemoryCapacity(self):
        return self.memoryCapacity

    def getParameters(self):
        return self.parameters

    def getNumOfActions(self):
        return self.num_actions

    def getEpsilon(self):
        return self.epsilon

    def getDiscount(self):
        return self.discount

    def getFrameSkipRate(self):
        return self.frameSkipRate

    def getGridSquaresPerFov(self):
        return self.gridSquaresPerFov

    def getTargetNetworkMaxSteps(self):
        return self.targetNetworkMaxSteps

    def getStateReprLen(self):
        return self.stateReprLen

    def getHiddenLayer1(self):
        return self.hiddenLayer1

    def getHiddenLayer2(self):
        return self.hiddenLayer2

    def getHiddenLayer3(self):
        return self.hiddenLayer3

    def getNumActions(self):
        return self.num_actions

    def getLearningRate(self):
        return self.learningRate

    def getActivationFuncHidden(self):
        return self.activationFuncHidden

    def getActivationFuncOutput(self):
        return self.activationFuncOutput

    def getOptimizer(self):
        return self.optimizer

    def getLoadedModelName(self):
        return self.loadedModelName

    def getActions(self):
        return self.actions

    def getTargetNetwork(self):
        return self.targetNetwork

    def getValueNetwork(self):
        return self.valueNetwork



