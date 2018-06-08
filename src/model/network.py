import heapq
import keras

keras.backend.set_image_dim_ordering('tf')
import numpy

numpy.set_printoptions(threshold=numpy.nan)
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LSTM, Softmax, Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Sequential, Model
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model, save_model

from .parameters import *

def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def createDiscreteActionsCircle(numActions, enableSplit, enableEject):
    actions = []
    # Add standing still action:
    # Add all other actions:
    degPerAction = 360 / numActions
    for degreeCount in range(numActions):
        degree = math.radians(degreeCount * degPerAction)
        x = math.cos(degree)
        y = math.sin(degree)
        action = [x, y, 0, 0]
        actions.append(action)
        if enableSplit:
            splitAction = [x, y, 1, 0]
            actions.append(splitAction)
        if enableEject:
            ejectAction = [x, y, 0, 1]
            actions.append(ejectAction)
    return actions


def createDiscreteActionsSquare(numActions, enableSplit, enableEject):
    if not is_square(numActions):
        print("Number of Actions has to be a perfect square for this mode.")
        quit()
    actionsPerSide = int(math.sqrt(numActions))

    actions = []
    for row in range(actionsPerSide):
        for col in range(actionsPerSide):
            x = col / actionsPerSide + 1 / actionsPerSide / 2
            y = row / actionsPerSide + 1 / actionsPerSide / 2
            action = [x, y, 0, 0]
            actions.append(action)
            if enableSplit:
                splitAction = [x, y, 1, 0]
                actions.append(splitAction)
            if enableEject:
                ejectAction = [x, y, 0, 1]
                actions.append(ejectAction)
    return actions


class Network(object):

    def __init__(self, trainMode, modelName, parameters, loadModel):
        self.parameters = parameters

        self.trainMode = trainMode


        if parameters.SQUARE_ACTIONS:
            self.actions = createDiscreteActionsSquare(self.parameters.NUM_ACTIONS, self.parameters.ENABLE_SPLIT,
                                                 self.parameters.ENABLE_EJECT)
        else:
            self.actions = createDiscreteActionsCircle(self.parameters.NUM_ACTIONS, self.parameters.ENABLE_SPLIT,
                                                       self.parameters.ENABLE_EJECT)
        self.num_actions = len(self.actions)

        self.loadedModelName = None

        self.gpus = self.parameters.GPUS

        # Q-learning
        self.discount = self.parameters.DISCOUNT
        self.epsilon = self.parameters.EPSILON
        self.frameSkipRate = self.parameters.FRAME_SKIP_RATE
        self.gridSquaresPerFov = self.parameters.GRID_SQUARES_PER_FOV

        # CNN
        if self.parameters.CNN_REPRESENTATION:
            self.kernelLen_1 = self.parameters.CNN_LAYER_1
            self.stride_1 = self.parameters.CNN_LAYER_1_STRIDE
            self.filterNum_1 = self.parameters.CNN_LAYER_1_FILTER_NUM

            self.kernelLen_2 = self.parameters.CNN_LAYER_2
            self.stride_2 = self.parameters.CNN_LAYER_2_STRIDE
            self.filterNum_2 = self.parameters.CNN_LAYER_2_FILTER_NUM

            self.kernelLen_3 = self.parameters.CNN_LAYER_3
            self.stride_3 = self.parameters.CNN_LAYER_3_STRIDE
            self.filterNum_3 = self.parameters.CNN_LAYER_3_FILTER_NUM

            if self.parameters.CNN_USE_LAYER_1:
                self.stateReprLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_1
            elif self.parameters.CNN_USE_LAYER_2:
                self.stateReprLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_2
            else:
                self.stateReprLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_3
        else:
            self.stateReprLen = self.parameters.STATE_REPR_LEN

        # ANN
        self.learningRate = self.parameters.ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        if self.parameters.ACTIVATION_FUNC_HIDDEN == "elu":
            self.activationFuncHidden = "linear"  # keras.layers.ELU(alpha=eluAlpha)
        else:
            self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncLSTM = self.parameters.ACTIVATION_FUNC_LSTM
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3

        if self.parameters.USE_ACTION_AS_INPUT:
            inputDim = self.stateReprLen + 4
            outputDim = 1
        else:
            inputDim = self.stateReprLen
            outputDim = self.num_actions

        if self.parameters.EXP_REPLAY_ENABLED:
            input_shape_lstm = (self.parameters.MEMORY_TRACE_LEN, inputDim)
            stateful_training = False
            self.batch_len = self.parameters.MEMORY_BATCH_LEN

        else:
            input_shape_lstm = (1, inputDim)
            stateful_training = True
            self.batch_len = 1

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + self.num_actions))
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                           maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.valueNetwork = Sequential()
                self.valueNetwork.add(Dense(self.hiddenLayer1, input_dim=inputDim, activation=self.activationFuncHidden,
                                            bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.valueNetwork.add(
                        Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.valueNetwork.add(
                        Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                self.valueNetwork.add(
                    Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                          , kernel_initializer=initializer))
                self.valueNetwork = multi_gpu_model(self.valueNetwork, gpus=self.gpus)
        else:
            # self.valueNetwork = Sequential()

            # CNN
            if self.parameters.CNN_REPRESENTATION:
                if self.parameters.CNN_PIXEL_REPRESENTATION:
                    if self.parameters.CNN_USE_LAYER_1:
                        inputLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_1
                    elif self.parameters.CNN_USE_LAYER_2:
                        inputLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_2
                    else:
                        inputLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_3


                    self.input = Input(shape=(inputLen, inputLen, 3))

                    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(self.input)
                    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

                    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(self.input)
                    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

                    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(self.input)
                    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

                    self.valueNetwork = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

                # Not pixel input
                else:
                    tower = []
                    self.input = []
                    self.towerModel = []
                    for grid in range(self.parameters.NUM_OF_GRIDS):
                        if self.parameters.CNN_USE_LAYER_1:
                            inputLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_1
                            self.input.append(Input(shape=(1, inputLen, inputLen)))
                            tower.append(Conv2D(self.filterNum_1, kernel_size=(self.kernelLen_1, self.kernelLen_1),
                                                strides=(self.stride_1, self.stride_1), activation='relu',
                                                data_format='channels_first')(self.input[grid]))

                        if self.parameters.CNN_USE_LAYER_2:
                            if self.parameters.CNN_USE_LAYER_1:
                                tower[grid] = Conv2D(self.filterNum_2, kernel_size=(self.kernelLen_2, self.kernelLen_2),
                                                     strides=(self.stride_2, self.stride_2), activation='relu',
                                                     data_format='channels_first')(tower[grid])
                            else:
                                inputLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_2
                                self.input.append(Input(shape=(1, inputLen, inputLen)))
                                tower.append(Conv2D(self.filterNum_2, kernel_size=(self.kernelLen_2, self.kernelLen_2),
                                                    strides=(self.stride_2, self.stride_2), activation='relu',
                                                    data_format='channels_first')(self.input[grid]))

                        if self.parameters.CNN_USE_LAYER_2:
                            tower[grid] = Conv2D(self.filterNum_3, kernel_size=(self.kernelLen_3, self.kernelLen_3),
                                                 strides=(self.stride_3, self.stride_3), activation='relu',
                                                 data_format='channels_first')(tower[grid])
                        else:
                            inputLen = self.parameters.CNN_SIZE_OF_INPUT_DIM_3
                            self.input.append(Input(shape=(1, inputLen, inputLen)))
                            tower.append(Conv2D(self.filterNum_3, kernel_size=(self.kernelLen_3, self.kernelLen_3),
                                                strides=(self.stride_3, self.stride_3), activation='relu',
                                                data_format='channels_first')(self.input[grid]))
                        tower[grid] = Flatten()(tower[grid])
                        # self.towerModel.append(Model(self.input[grid], tower[grid]))

                    self.valueNetwork = keras.layers.concatenate([i for i in tower], axis=1)

            # Fully connected layers
            if self.parameters.NEURON_TYPE == "MLP":
                # Hidden Layer 1
                if self.parameters.CNN_REPRESENTATION:
                    dense_layer = Dense(self.hiddenLayer1, activation=self.activationFuncHidden,
                                        bias_initializer=initializer, kernel_initializer=initializer)(self.valueNetwork)
                else:

                    self.input = Input(shape=(self.stateReprLen,))
                    dense_layer = Dense(self.hiddenLayer1, activation=self.activationFuncHidden,
                                        bias_initializer=initializer, kernel_initializer=initializer)(self.input)
                if self.parameters.ACTIVATION_FUNC_HIDDEN == "elu":
                    dense_layer = (keras.layers.ELU(alpha=self.parameters.ELU_ALPHA))(dense_layer)
                # Hidden 2
                if self.hiddenLayer2 > 0:
                    dense_layer = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                        bias_initializer=initializer, kernel_initializer=initializer)(dense_layer)
                    # self.valueNetwork.add(hidden2)
                    if self.parameters.ACTIVATION_FUNC_HIDDEN == "elu":
                        dense_layer = (keras.layers.ELU(alpha=self.parameters.ELU_ALPHA))(dense_layer)
                # Hidden 3
                if self.hiddenLayer3 > 0:
                    dense_layer = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                        bias_initializer=initializer, kernel_initializer=initializer)(dense_layer)
                    # self.valueNetwork.add(hidden3)
                    if self.parameters.ACTIVATION_FUNC_HIDDEN == "elu":
                        dense_layer = (keras.layers.ELU(alpha=self.parameters.ELU_ALPHA))(dense_layer)
                # Output layer
                self.output = Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                                    , kernel_initializer=initializer)(dense_layer)

                # Define functional model
                # if self.parameters.CNN_REPRESENTATION:
                    # input_shape = [i.input for i in self.towerModel]
                    # print(input_shape)
                self.valueNetwork = Model(inputs=self.input, outputs=self.output)
                # self.valueNetwork.add(output)

            elif self.parameters.NEURON_TYPE == "LSTM":
                # Hidden Layer 1
                # TODO: Use CNN with LSTM
                # if self.parameters.CNN_REPRESENTATION:
                #     hidden1 = LSTM(self.hiddenLayer1, return_sequences=True, stateful=stateful_training, batch_size=self.batch_len)
                # else:
                #     hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm, return_sequences = True,
                #                    stateful= stateful_training, batch_size=self.batch_len)
                hidden1 = LSTM(self.hiddenLayer1, input_shape=input_shape_lstm, return_sequences=True,
                               stateful=stateful_training, batch_size=self.batch_len, bias_initializer=initializer
                               , kernel_initializer=initializer)
                self.valueNetwork.add(hidden1)
                # Hidden 2
                if self.hiddenLayer2 > 0:
                    hidden2 = LSTM(self.hiddenLayer2, return_sequences=True, stateful=stateful_training,
                                   batch_size=self.batch_len, bias_initializer=initializer
                                   , kernel_initializer=initializer)
                    self.valueNetwork.add(hidden2)
                # Hidden 3
                if self.hiddenLayer3 > 0:
                    hidden3 = LSTM(self.hiddenLayer3, return_sequences=True, stateful=stateful_training,
                                   batch_size=self.batch_len, bias_initializer=initializer
                                   , kernel_initializer=initializer)
                    self.valueNetwork.add(hidden3)
                # Output layer
                output = LSTM(outputDim, activation=self.activationFuncOutput,
                              return_sequences=True, stateful=stateful_training, batch_size=self.batch_len,
                              bias_initializer=initializer
                              , kernel_initializer=initializer)
                self.valueNetwork.add(output)

        # Create target network
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
                           return_sequences=True, stateful=True, batch_size=1, bias_initializer=initializer
                           , kernel_initializer=initializer)
            self.actionNetwork.add(hidden1)

            if self.hiddenLayer2 > 0:
                hidden2 = LSTM(self.hiddenLayer2, return_sequences=True, stateful=True, batch_size=self.batch_len,
                               bias_initializer=initializer, kernel_initializer=initializer)
                self.actionNetwork.add(hidden2)
            if self.hiddenLayer3 > 0:
                hidden3 = LSTM(self.hiddenLayer3, return_sequences=True, stateful=True, batch_size=self.batch_len,
                               bias_initializer=initializer, kernel_initializer=initializer)
                self.actionNetwork.add(hidden3)
            self.actionNetwork.add(LSTM(self.num_actions, activation=self.activationFuncOutput,
                                        return_sequences=False, stateful=True, batch_size=self.batch_len,
                                        bias_initializer=initializer, kernel_initializer=initializer))
            self.actionNetwork.compile(loss='mse', optimizer=optimizer)

        print(self.valueNetwork.summary())
        if loadModel:
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
        self.valueNetwork = keras.models.load_model(path + "model.h5")
        self.targetNetwork = load_model(path + "model.h5")

    def trainOnBatch(self, inputs, targets):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.valueNetwork.train_on_batch(inputs, targets)
            else:
                return self.valueNetwork.train_on_batch(numpy.array([numpy.array([inputs])]),
                                                        numpy.array([numpy.array([targets])]))
        else:
            return self.valueNetwork.train_on_batch(inputs, targets)

    def updateActionNetwork(self):
        self.actionNetwork.set_weights(self.valueNetwork.get_weights())

    def updateTargetNetwork(self):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())

    def predict(self, state, batch_len=1):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.valueNetwork.predict(state, batch_size=batch_len)
            else:
                return self.valueNetwork.predict(numpy.array([numpy.array([state])]))[0][0]
        if self.parameters.CNN_REPRESENTATION:

            stateRepr = numpy.zeros((len(state), 1, 1,  len(state[0]), len(state[0])))

            for gridIdx, grid in enumerate(state):
                stateRepr[gridIdx][0][0] = grid
            stateRepr = list(stateRepr)

            return self.valueNetwork.predict(stateRepr)[0]
        else:
            return self.valueNetwork.predict(state)[0]

    def predictTargetQValues(self, state):
        if self.parameters.USE_ACTION_AS_INPUT:
            return [self.predict_target_network(numpy.concatenate((state, act))) for act in self.actions]
        else:
            return self.predict_target_network(state)

    def predict_target_network(self, state, len_batch=1):
        if self.parameters.NEURON_TYPE == "LSTM":
            if self.parameters.EXP_REPLAY_ENABLED:
                return self.targetNetwork.predict(state, batch_size=len_batch)
            else:
                return self.targetNetwork.predict(numpy.array([numpy.array([state])]))[0][0]
        if self.parameters.CNN_REPRESENTATION:
            stateRepr = numpy.zeros((len(state), 1, 1,  len(state[0]), len(state[0])))

            for gridIdx, grid in enumerate(state):
                stateRepr[gridIdx][0][0] = grid
            stateRepr = list(stateRepr)

            return self.valueNetwork.predict(stateRepr)[0]
        else:
            return self.targetNetwork.predict(state)[0]


    def predict_action_network(self, trace):
        return self.actionNetwork.predict(numpy.array([numpy.array([trace])]))[0]

    def predict_action(self, state):
        if self.parameters.USE_ACTION_AS_INPUT:
            return [self.predict(numpy.concatenate((state, act))) for act in self.actions]
        else:
            if self.parameters.NEURON_TYPE == "MLP":
                return self.predict(state)
            else:
                return self.predict_action_network(state)

    def saveModel(self, path, name=""):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())
        self.targetNetwork.save(path + name + "model.h5")

    def setEpsilon(self, val):
        self.epsilon = val

    def setFrameSkipRate(self, value):
        self.frameSkipRate = value

    def getTrainMode(self):
        return self.trainMode

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
