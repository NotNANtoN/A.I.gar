import heapq
import keras
import numpy
import tensorflow as tf
import importlib.util
from keras.layers import Dense, LSTM, Softmax
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model


from .parameters import *

class Network(object):

    def __init__(self, trainMode, modelName):
        self.trainMode = trainMode

        self.actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
                   eject in [0, 1]]
        # Filter out actions that do a split and eject at the same time
        # Also filter eject actions for now
        for action in self.actions[:]:
            if action[2] or action[3]:
                self.actions.remove(action)

        self.num_actions = len(self.actions)
        self.parameters = importlib.import_module('.networkParameters', package="model")
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
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3

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
                hidden1 = LSTM(self.hiddenLayer1, input_shape=(self.stateReprLen, 1), activation=self.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)

            self.valueNetwork.add(hidden1)
            #self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.valueNetwork.add(hidden2)
                #self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.valueNetwork.add(hidden3)
                #self.valueNetwork.add(Dropout(0.5))

            self.valueNetwork.add(
                Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                      , kernel_initializer=initializer))


        self.targetNetwork = keras.models.clone_model(self.valueNetwork)
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())



        if self.optimizer == "Adam":
            optimizer = keras.optimizers.Adam(lr=self.learningRate)
        else:
            optimizer = keras.optimizers.SGD(lr=self.learningRate)

        self.valueNetwork.compile(loss='mse', optimizer=optimizer)
        self.targetNetwork.compile(loss='mse', optimizer=optimizer)


        if modelName is not None:
            self.load(modelName)

    def load(self, modelName):
        path = "savedModels/" + modelName
        packageName = "savedModels." + modelName
        self.parameters = importlib.import_module('.networkParameters', package=packageName)
        self.loadedModelName = modelName
        self.valueNetwork = load_model(path + "/NN_model.h5")
        self.targetNetwork = load_model(path + "/NN_model.h5")

    def trainOnBatch(self, inputs, targets):
        self.valueNetwork.train_on_batch(inputs, targets)

    def predict(self, state):
        return self.valueNetwork.predict(numpy.array([state]))[0]

    def predict_target_network(self, state):
        return self.targetNetwork.predict(numpy.array([state]))[0]

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



