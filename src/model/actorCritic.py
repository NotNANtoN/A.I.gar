import keras
import numpy
import math
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model
from keras import backend as K
from keras.layers import Conv2D, Flatten, Input, Dense
from keras.models import Model

def relu_max(x):
    return K.relu(x, max_value=1)

class ValueNetwork(object):
    def __init__(self, parameters, modelName, cnnLayers=None, cnnInput=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.learningRate = self.parameters.CACLA_CRITIC_ALPHA
        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.layers = parameters.CACLA_CRITIC_LAYERS
        self.input = None

        if modelName is not None:
            self.load(modelName)
            return

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + 1))
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                           maxval=weight_initializer_range, seed=None)

        regularizer = keras.regularizers.l2(self.parameters.CACLA_CRITIC_WEIGHT_DECAY)
        layerIterable = iter(self.layers)

        if self.parameters.CNN_REPR:
            self.input = cnnInput
            previousLayer = cnnLayers
            extraInputSize = self.parameters.EXTRA_INPUT
            if extraInputSize > 0:
                extraInput = Input(shape=(extraInputSize,))
                self.input = [cnnInput, extraInput]
                denseInput = keras.layers.concatenate([cnnLayers, extraInput])
                previousLayer = Dense(next(layerIterable), activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer,
                                      kernel_regularizer=regularizer)(denseInput)
        else:
            self.input = keras.layers.Input((self.stateReprLen,))
            previousLayer = self.input

        for layer in layerIterable:
            if layer > 0:
                previousLayer = Dense(layer, activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer,
                                      kernel_regularizer=regularizer)(previousLayer)
                if self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY == "elu":
                    previousLayer = (keras.layers.ELU(alpha=self.parameters.ELU_ALPHA))(previousLayer)

        output = Dense(1, activation="linear", bias_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=regularizer)(previousLayer)

        self.model = Model(inputs=self.input, outputs=output)


        optimizer = keras.optimizers.Adam(lr=self.learningRate, amsgrad=self.parameters.AMSGRAD)

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "value_model.h5")
            self.target_model = load_model(path + "value_model.h5")

    def predict(self, state):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.model.predict(state)[0][0]

    def predict_target_model(self, state):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.target_model.predict(state)[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            tau = self.parameters.DPG_TAU
        else:
            tau = self.parameters.CACLA_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets, importance_weights):
        self.model.train_on_batch(inputs, targets, sample_weight=importance_weights)


    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "value_model.h5")


class PolicyNetwork(object):
    def __init__(self, parameters, modelName, cnnLayers=None, cnnInput=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.input = None


        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.learningRate = self.parameters.DPG_ACTOR_ALPHA
            self.layers = parameters.DPG_ACTOR_LAYERS
        else:
            self.learningRate = self.parameters.CACLA_ACTOR_ALPHA
            self.layers = parameters.CACLA_ACTOR_LAYERS

        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY


        self.num_outputs = 2  #x, y, split, eject all continuous between 0 and 1
        if self.parameters.ENABLE_SPLIT:
            self.num_outputs += 1
        if self.parameters.ENABLE_EJECT:
            self.num_outputs += 1

        if modelName is not None:
            self.load(modelName)
            return

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + 1))
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                           maxval=weight_initializer_range, seed=None)

        layerIterable = iter(self.layers)

        if self.parameters.CNN_REPR:
            self.input = cnnInput
            previousLayer = cnnLayers
            extraInputSize = self.parameters.EXTRA_INPUT
            if extraInputSize > 0:
                extraInput = Input(shape=(extraInputSize,))
                self.input = [cnnInput, extraInput]
                denseInput = keras.layers.concatenate([cnnLayers, extraInput])
                previousLayer = Dense(next(layerIterable), activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer)(denseInput)
        else:
            self.input = keras.layers.Input((self.stateReprLen,))
            previousLayer = self.input

        for neuronNumber in layerIterable:
            if neuronNumber > 0:
                previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden, bias_initializer=initializer,
                                      kernel_initializer=initializer)(previousLayer)


        output = Dense(self.num_outputs, activation="sigmoid", bias_initializer=initializer,
                       kernel_initializer=initializer)(previousLayer)
        self.model = keras.models.Model(inputs=self.input, outputs=output)

        optimizer = keras.optimizers.Adam(lr=self.learningRate, amsgrad=self.parameters.AMSGRAD)

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)


    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "actor_model.h5")

    def predict(self, state):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.model.predict(state)[0]

    def predict_target_model(self, state):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])

                state = [grid, extra]
            else:
                state = numpy.array([state])
        return self.target_model.predict(state)


    def train(self, inputs, targets, weights = None):
        if self.parameters.ACTOR_IS and weights is not None:
            self.model.train_on_batch(inputs, targets, sample_weight=weights)
        else:
            self.model.train_on_batch(inputs, targets)

    def update_target_model(self):
        if self.parameters.ACTOR_CRITIC_TYPE != "DPG":
            return
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        if self.parameters.ACTOR_CRITIC_TYPE != "DPG":
            return
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def save(self, path, name = ""):
        self.model.save(path + name + "actor" + "_model.h5")


class ActionValueNetwork(object):
    def __init__(self, parameters, modelName, cnnLayers=None, cnnInput=None):
        self.ornUhlPrev = 0
        self.parameters = parameters
        self.loadedModelName = None
        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.learningRate = self.parameters.DPG_CRITIC_ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        self.activationFuncHidden = self.parameters.DPG_CRITIC_FUNC
        self.layers = self.parameters.DPG_CRITIC_LAYERS

        self.num_actions_inputs = 2  # x, y, split, eject all continuous between 0 and 1
        if self.parameters.ENABLE_SPLIT:
            self.num_actions_inputs += 1
        if self.parameters.ENABLE_EJECT:
            self.num_actions_inputs += 1

        if modelName is not None:
            self.load(modelName)
            return

        initializer = keras.initializers.glorot_uniform()
        regularizer = keras.regularizers.l2(self.parameters.DPG_CRITIC_WEIGHT_DECAY)

        layerIterable = enumerate(self.layers)
        self.inputAction = keras.layers.Input((self.num_actions_inputs,))

        if self.parameters.CNN_REPR:
            self.inputState = cnnInput
            previousLayer = cnnLayers
            extraInputSize = self.parameters.EXTRA_INPUT
            if extraInputSize > 0:
                idx, neuronNumber = next(layerIterable)
                extraInput = Input(shape=(extraInputSize,))
                self.inputState = [cnnInput, extraInput]
                denseInput = keras.layers.concatenate([cnnLayers, extraInput])
                if idx == parameters.DPG_FEED_ACTION_IN_LAYER - 1:
                    mergeLayer = keras.layers.concatenate([denseInput, self.inputAction])
                    denseInput = mergeLayer
                previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden,
                                      bias_initializer=initializer, kernel_initializer=initializer)(denseInput)
        else:
            self.inputState = keras.layers.Input((self.stateReprLen,))
            previousLayer = self.inputState

        for idx, neuronNumber in layerIterable:
            if idx == parameters.DPG_FEED_ACTION_IN_LAYER - 1:
                mergeLayer = keras.layers.concatenate([previousLayer, self.inputAction])
                previousLayer = mergeLayer
            previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden, bias_initializer=initializer,
                                  kernel_initializer=initializer, kernel_regularizer=regularizer)(previousLayer)

        output = Dense(1, activation="linear", bias_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=regularizer)(previousLayer)
        if self.parameters.CNN_REPR:
            extraInputSize = self.parameters.EXTRA_INPUT
            if extraInputSize > 0:
                self.model = keras.models.Model(inputs=[self.inputState[0], self.inputState[1], self.inputAction], outputs=output)
            else:
                self.model = keras.models.Model(inputs=[self.inputState, self.inputAction], outputs=output)
        else:
            self.model = keras.models.Model(inputs=[self.inputState, self.inputAction], outputs=output)


        optimizer = keras.optimizers.Adam(lr=self.learningRate, amsgrad=self.parameters.AMSGRAD)

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "actionValue_model.h5")
            self.target_model = load_model(path + "actionValue_model.h5")

    def predict(self, state, action):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])
                return self.target_model.predict([grid, extra, action])[0][0]
            else:
                grid = numpy.array([state])
                return self.target_model.predict([grid, action])[0][0]

        return self.model.predict([state, action])[0][0]

    def predict_target_model(self, state, action):
        if self.parameters.CNN_REPR:
            if len(state) == 2:
                grid = numpy.array([state[0]])
                extra = numpy.array([state[1]])
                #
                # state = [grid, extra]
                return self.target_model.predict([grid, extra, action])[0][0]

            else:
                grid = numpy.array([state])
                return self.target_model.predict([grid, action])[0][0]

            # action = numpy.array([action])

        return self.target_model.predict([state, action])[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets, importance_weights):
        self.model.train_on_batch(inputs, targets, sample_weight=importance_weights)


    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "actionValue_model.h5")


class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters):
        self.discount = 0 if parameters.END_DISCOUNT else parameters.DISCOUNT
        self.discrete = False
        self.acType = parameters.ACTOR_CRITIC_TYPE
        self.parameters = parameters
        self.std = self.parameters.GAUSSIAN_NOISE
        self.noise_decay_factor = self.parameters.AC_NOISE_DECAY
        self.ocacla_noise = 1
        self.ocacla_noise_decay = self.parameters.OCACLA_NOISE_DECAY
        self.steps = 0
        self.action_len = 2 + self.parameters.ENABLE_SPLIT + self.parameters.ENABLE_EJECT
        self.ornUhlPrev = numpy.zeros(self.action_len)
        self.counts = [] # For SPG/CACLA: count how much actor training we do each step
        self.caclaVar = parameters.CACLA_VAR_START
        self.input_len = parameters.STATE_REPR_LEN
        self.input = None

        # CNN stuff:
        if self.parameters.CNN_REPR:
            if self.parameters.CNN_P_REPR:
                if self.parameters.CNN_P_RGB:
                    channels = 3
                # GrayScale
                else:
                    channels = 1
                if self.parameters.CNN_LAST_GRID:
                    channels = channels * 2
                if self.parameters.COORDCONV:
                    channels += 2

                if self.parameters.CNN_USE_L1:
                    self.input_len = (self.parameters.CNN_INPUT_DIM_1,
                                      self.parameters.CNN_INPUT_DIM_1, channels)
                elif self.parameters.CNN_USE_L2:
                    self.input_len = (self.parameters.CNN_INPUT_DIM_2,
                                      self.parameters.CNN_INPUT_DIM_2, channels)
                else:
                    self.input_len = (self.parameters.CNN_INPUT_DIM_3,
                                      self.parameters.CNN_INPUT_DIM_3, channels)
            else:
                channels = self.parameters.NUM_OF_GRIDS
                if self.parameters.CNN_USE_L1:
                    self.input_len = (channels, self.parameters.CNN_INPUT_DIM_1,
                                      self.parameters.CNN_INPUT_DIM_1)
                elif self.parameters.CNN_USE_L2:
                    self.input_len = (channels, self.parameters.CNN_INPUT_DIM_2,
                                      self.parameters.CNN_INPUT_DIM_2)
                else:
                    self.input_len = (channels, self.parameters.CNN_INPUT_DIM_3,
                                      self.parameters.CNN_INPUT_DIM_3)
            if self.parameters.EXTRA_INPUT:
                self.input_len = [self.input_len, self.parameters.EXTRA_INPUT]

        # Bookkeeping:
        self.latestTDerror = None
        self.qValues = []
        self.actor = None
        self.critic = None
        self.combinedActorCritic = None

    def createCombinedActorCritic(self, actor, critic):
        for layer in critic.model.layers:
            layer.trainable = False
        #mergeLayer = keras.layers.concatenate([actor.inputs[0], actor.outputs[0]])
        nonTrainableCritic = critic.model([actor.model.inputs[0], actor.model.outputs[0]])
        combinedModel = keras.models.Model(inputs=actor.model.inputs, outputs=nonTrainableCritic)
        if self.parameters.DPG_ACTOR_OPTIMIZER == "Adam":
            optimizer = keras.optimizers.Adam(lr=actor.learningRate, amsgrad=self.parameters.AMSGRAD)
        elif self.parameters.DPG_ACTOR_OPTIMIZER == "SGD":
            if self.parameters.DPG_ACTOR_NESTEROV:
                optimizer = keras.optimizers.SGD(lr=actor.learningRate,momentum=self.parameters.DPG_ACTOR_NESTEROV,
                                                 nesterov=True)
            else:
                optimizer = keras.optimizers.SGD(lr=actor.learningRate)


        combinedModel.compile(optimizer=optimizer, loss="mse")
        return combinedModel

    def initializeNetwork(self, loadPath, networks=None):
        if networks is None or networks == {}:
            if networks is None:
                networks = {}
            cnnLayers = None
            if self.parameters.CNN_REPR:
                cnnLayers = self.createCNN()
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                self.actor = PolicyNetwork(self.parameters, loadPath)
                self.critic = ActionValueNetwork(self.parameters, loadPath)
                self.combinedActorCritic = self.createCombinedActorCritic(self.actor, self.critic)
                networks["MU(S)"] = self.actor
                networks["Q(S,A)"] = self.critic
                networks["Actor-Critic-Combo"] = self.combinedActorCritic
            else:
                self.actor = PolicyNetwork(self.parameters, loadPath, cnnLayers, self.input)
                networks["MU(S)"] = self.actor
                if self.parameters.OCACLA_ENABLED:
                    self.critic = ActionValueNetwork(self.parameters, loadPath, cnnLayers, self.input)
                    networks["Q(S,A)"] = self.critic
                else:
                    self.critic = ValueNetwork(self.parameters, loadPath, cnnLayers, self.input)
                    networks["V(S)"] = self.critic
        else:
            self.actor  = networks["MU(S)"]
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                self.critic = networks["Q(S,A)"]
                self.combinedActorCritic = networks["Actor-Critic-Combo"]
            else:
                if self.parameters.OCACLA_ENABLED:
                    self.critic = networks["Q(S,A)"]
                else:
                    self.critic = networks["V(S)"]
        for network in networks:
            print(network + " summary:")
            if network == "Actor-Critic-Combo":
                networks[network].summary()
                continue
            networks[network].model.summary()
        return networks

    def createCNN(self):
        # (KernelSize, stride, filterNum)
        kernel_1 = self.parameters.CNN_L1

        kernel_2 = self.parameters.CNN_L2

        kernel_3 = self.parameters.CNN_L3

        # Pixel input
        if self.parameters.CNN_P_REPR:
            data_format = 'channels_last'
        # Not pixel input
        else:
            data_format = 'channels_first'

        if self.parameters.EXTRA_INPUT:
            input_len = self.input_len[0]
        else:
            input_len = self.input_len

        self.input = Input(shape=input_len)
        conv = self.input
        if self.parameters.CNN_USE_L1:
            conv = Conv2D(kernel_1[2], kernel_size=(kernel_1[0], kernel_1[0]),
                          strides=(kernel_1[1], kernel_1[1]), activation='relu',
                          data_format=data_format)(conv)
        if self.parameters.CNN_USE_L2:
            conv = Conv2D(kernel_2[2], kernel_size=(kernel_2[0], kernel_2[0]),
                          strides=(kernel_2[1], kernel_2[1]), activation='relu',
                          data_format=data_format)(conv)
        if self.parameters.CNN_USE_L3:
            conv = Conv2D(kernel_3[2], kernel_size=(kernel_3[0], kernel_3[0]),
                          strides=(kernel_3[1], kernel_3[1]), activation='relu',
                          data_format=data_format)(conv)

        cnnLayers = Flatten()(conv)
        return cnnLayers

    def updateNoise(self):
        self.std *= self.noise_decay_factor
        self.ocacla_noise *= self.ocacla_noise_decay
        if self.parameters.END_DISCOUNT:
            self.discount = 1 - self.parameters.DISCOUNT_INCREASE_FACTOR * (1 - self.discount)

    def updateCriticNetworks(self, time):
        if time % self.parameters.TARGET_NETWORK_STEPS == 0:
            self.critic.update_target_model()
            self.actor.update_target_model()

    def softlyUpdateNetworks(self):
        self.actor.softlyUpdateTargetModel()
        self.critic.softlyUpdateTargetModel()

    def updateNetworks(self, time):
        if self.parameters.SOFT_TARGET_UPDATES:
            self.softlyUpdateNetworks()
        else:
            self.updateCriticNetworks(time)

    def apply_off_policy_corrections_cacla(self, batch):
        batchLen = len(batch[0])
        #if not self.parameters.CACLA_OFF_POLICY_CORR:
        #    return numpy.ones(batchLen)

        off_policy_weights = []
        for idx in range(batchLen):
            state = batch[0][idx]
            action = batch[1][idx]
            behavior_action = batch[4][idx]
            action_current_policy = self.actor.predict(state)

            policy_difference = action_current_policy - behavior_action

            squared_sum = policy_difference[0] ** 2 + policy_difference[1] ** 2
            if self.parameters.ENABLE_SPLIT or self.parameters.ENABLE_EJECT:
                squared_sum += policy_difference[2] ** 2
            if self.parameters.ENABLE_SPLIT and self.parameters.ENABLE_EJECT:
                squared_sum += policy_difference[3] ** 2
            magnitude_difference = math.sqrt(squared_sum)


            off_policy_correction = 1 / ((1 + magnitude_difference) ** self.parameters.CACLA_OFF_POLICY_CORR)

            if self.parameters.CACLA_OFF_POLICY_CORR_SIGN:
                behavior_vector = action - behavior_action
                current_policy_vector = action - action_current_policy
                dot_prod = numpy.dot(behavior_vector, current_policy_vector)
                if dot_prod < 0:
                    off_policy_correction = 0
                #TODO: make more sophisticated by making it scale depending on angle: 0deg is 1, 90 deg is 0

            off_policy_weights.append(off_policy_correction)

        return off_policy_weights

    def learn(self, batch, steps):
        updated_actions = None
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            idxs, priorities = self.train_critic_DPG(batch)
            if self.parameters.DPG_USE_DPG_ACTOR_TRAINING and steps > self.parameters.AC_ACTOR_TRAINING_START and \
                    (steps > self.parameters.DPG_CACLA_STEPS or steps <= self.parameters.DPG_DPG_STEPS):
                self.train_actor_DPG(batch)
            if (self.parameters.DPG_USE_CACLA or steps < self.parameters.DPG_CACLA_STEPS\
                    or steps > self.parameters.DPG_DPG_STEPS) and steps > self.parameters.AC_ACTOR_TRAINING_START:
                priorities = self.train_actor_batch(batch, priorities)
        else:
            if self.parameters.OCACLA_ENABLED:
                idxs, priorities = self.train_critic_DPG(batch, get_evals=True)
                if steps > self.parameters.AC_ACTOR_TRAINING_START:
                    updated_actions = self.train_actor_OCACLA(batch, priorities)
            else:
                off_policy_weights = self.apply_off_policy_corrections_cacla(batch)
                idxs, priorities = self.train_critic(batch, off_policy_weights)
                if steps > self.parameters.AC_ACTOR_TRAINING_START:
                    priorities = self.train_actor_batch(batch, priorities, off_policy_weights)
        self.latestTDerror = numpy.mean(priorities)
        return idxs, priorities, updated_actions

    def train_actor_DPG(self, batch):
        batch_len = len(batch[0])
        inputs = numpy.zeros((batch_len, self.parameters.STATE_REPR_LEN))
        targets = numpy.zeros((batch_len, 1))
        importance_weights = batch[5] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else numpy.ones(batch_len)


        # Calculate input and target for actor
        for sample_idx in range(batch_len):
            old_s, a, r, new_s = batch[0][sample_idx], batch[1][sample_idx], batch[2][sample_idx], batch[3][
                sample_idx]
            oldPrediction = self.combinedActorCritic.predict(old_s)[0]
            inputs[sample_idx] = old_s
            targets[sample_idx] = oldPrediction + self.parameters.DPG_Q_VAL_INCREASE

        if self.parameters.ACTOR_IS:
            self.combinedActorCritic.train_on_batch(inputs, targets, sample_weight=importance_weights)
        else:
            self.combinedActorCritic.train_on_batch(inputs, targets)

    def train_actor_batch(self, batch, priorities, off_policy_weights = None):
        batch_len = len(batch[0])
        len_output = self.actor.num_outputs
        if self.parameters.CNN_REPR:
            inputShape = numpy.array([batch_len] + list(self.input_len[0]))
            inputs = numpy.zeros(inputShape)
            if self.parameters.EXTRA_INPUT:
                extraInput = numpy.zeros((batch_len, self.input_len[1]))
                inputs = [inputs, extraInput]
        else:
            inputs = numpy.zeros((batch_len, self.input_len))
        targets = numpy.zeros((batch_len, len_output))
        used_imp_weights = numpy.zeros(batch_len)
        importance_weights = batch[5] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else numpy.ones(batch_len)
        train_count_cacla_var = numpy.zeros(batch_len)
        if off_policy_weights is not None:
            importance_weights *= off_policy_weights

        # Calculate input and target for actor
        pos_tde_count = 0
        for sample_idx in range(batch_len):
            old_s, a, r, new_s = batch[0][sample_idx], batch[1][sample_idx], batch[2][sample_idx], batch[3][sample_idx]
            sample_weight = importance_weights[sample_idx]
            td_e = priorities[sample_idx]
            if self.parameters.CACLA_VAR_ENABLED:
                beta = self.parameters.CACLA_VAR_BETA
                self.caclaVar = (1 - beta) * self.caclaVar + beta * (td_e ** 2)
                train_count_cacla_var[pos_tde_count] = math.ceil(td_e / math.sqrt(self.caclaVar))
            target = self.calculateTarget_Actor(old_s, a, td_e)

            if target is not None and sample_weight != 0:
                if self.parameters.CNN_REPR and self.parameters.EXTRA_INPUT:
                    inputs[0][pos_tde_count] = old_s[0]
                    inputs[1][pos_tde_count] = old_s[1]
                else:
                    inputs[pos_tde_count] = old_s
                targets[pos_tde_count] = target
                used_imp_weights[pos_tde_count] = sample_weight
                pos_tde_count += 1
                if self.parameters.AC_ACTOR_TDE:
                    current_action = self.actor.predict(old_s)
                    actor_TDE = (target[0] - current_action[0]) ** 2 + (target[1] - current_action[1]) ** 2
                    if self.parameters.ENABLE_SPLIT or self.parameters.ENABLE_EJECT:
                        actor_TDE += (target[2] - current_action[2]) ** 2
                        if self.parameters.ENABLE_SPLIT and self.parameters.ENABLE_EJECT:
                            actor_TDE += (target[3] - current_action[3]) ** 2
                    priorities[sample_idx] += math.sqrt(actor_TDE) * self.parameters.AC_ACTOR_TDE
        self.counts.append(pos_tde_count)
        if self.parameters.CACLA_VAR_ENABLED:
            if pos_tde_count > 0:
                maxEpochs = int(max(train_count_cacla_var))
                for epoch in range(maxEpochs):
                    training_this_epoch = 0
                    for idx_count, train_count in enumerate(train_count_cacla_var[:pos_tde_count]):
                        if train_count > 0:
                            train_count_cacla_var[idx_count] -= 1
                            inputs[training_this_epoch] = inputs[idx_count]
                            targets[training_this_epoch] = targets[idx_count]
                            used_imp_weights[training_this_epoch] = used_imp_weights[idx_count]
                            training_this_epoch += 1
                    trainInputs = inputs[:training_this_epoch]
                    trainTargets = targets[:training_this_epoch]
                    train_used_imp_weights = used_imp_weights[:training_this_epoch]
                    self.actor.train(trainInputs, trainTargets, train_used_imp_weights)
        else:
            if pos_tde_count > 0:
                if self.parameters.CNN_REPR and self.parameters.EXTRA_INPUT:
                    inputs[0] = inputs[0][:pos_tde_count]
                    inputs[1] = inputs[1][:pos_tde_count]
                else:
                    inputs = inputs[:pos_tde_count]
                targets = targets[:pos_tde_count]
                used_imp_weights = used_imp_weights[:pos_tde_count]
                self.actor.train(inputs, targets, used_imp_weights)

        return priorities

    def train_actor_OCACLA(self, batch, evals):
        batch_len = len(batch[0])
        len_output = self.actor.num_outputs

        inputs = numpy.zeros((batch_len, self.input_len))
        targets = numpy.zeros((batch_len, len_output))
        used_imp_weights = numpy.zeros(batch_len)
        updated_actions = batch[1][:]
        importance_weights = batch[5] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else numpy.ones(batch_len)

        count = 0
        for sample_idx in range(batch_len):
            old_s, a, idx, sample_a = batch[0][sample_idx], batch[1][sample_idx], batch[6][sample_idx], batch[4][sample_idx]
            sample_weight = importance_weights[sample_idx]
            eval = evals[sample_idx]
            current_policy_action = self.actor.predict(old_s)
            eval_of_current_policy = self.critic.predict(old_s, numpy.array([current_policy_action]))
            best_action = a
            best_action_eval = eval
            # Conduct offline exploration in action space:
            if self.parameters.OCACLA_EXPL_SAMPLES:
                if self.parameters.OCACLA_REPLACE_TRANSITIONS and sample_a is not None:
                    eval_sample_a = self.critic.predict(old_s, numpy.array([sample_a]))
                    if eval_sample_a > best_action_eval:
                        best_action_eval = eval_sample_a
                        best_action = sample_a
                if eval_of_current_policy > best_action_eval:
                    best_action_eval = eval_of_current_policy
                    best_action = current_policy_action
                for x in range(self.parameters.OCACLA_EXPL_SAMPLES):
                    if self.parameters.OCACLA_MOVING_GAUSSIAN:
                        noisy_sample_action = self.applyNoise(best_action, self.ocacla_noise)
                    else:
                        noisy_sample_action = self.applyNoise(current_policy_action, self.ocacla_noise)
                    eval_of_noisy_action = self.critic.predict(old_s, numpy.array([noisy_sample_action]))
                    if eval_of_noisy_action > best_action_eval:
                        best_action_eval = eval_of_noisy_action
                        best_action = noisy_sample_action
            # Check if the best sampled action is better than our current prediction
            if best_action_eval > eval_of_current_policy:
                inputs[count] = old_s
                targets[count] = best_action
                used_imp_weights[count] = sample_weight
                updated_actions[sample_idx] = best_action
                count += 1

        self.counts.append(count) # debug info
        if count > 0:
            inputs = inputs[:count]
            targets = targets[:count]
            used_imp_weights = used_imp_weights[:count]
            self.actor.train(inputs, targets, used_imp_weights)
        return updated_actions


    def applyNoise(self, action, std = None):
        if std is None:
            std = self.std
        #Gaussian Noise:
        if self.parameters.NOISE_TYPE == "Gaussian":
            action = [numpy.random.normal(output, std) for output in action]
        elif self.parameters.NOISE_TYPE == "Orn-Uhl":
            for idx in range(len(action)):
                noise = self.ornUhlPrev[idx] + self.parameters.ORN_UHL_THETA * (self.parameters.ORN_UHL_MU -
                                                                                self.ornUhlPrev[idx]) \
                        * self.parameters.ORN_UHL_DT + self.std * numpy.sqrt(self.parameters.ORN_UHL_DT) \
                        * numpy.random.normal()
                self.ornUhlPrev[idx] = noise
                action[idx] += noise
        return numpy.clip(action, 0, 1)

    def decideMove(self, state, bot):
        action = self.actor.predict(state)

        if self.parameters.OCACLA_ONLINE_SAMPLES:
            action_eval = self.critic.predict(state, numpy.array([action]))
            for sample_idx in range(self.parameters.OCACLA_ONLINE_SAMPLES):
                noisyAction = self.applyNoise(action)
                noisy_eval = self.critic.predict(state, numpy.array([noisyAction]))
                if noisy_eval > action_eval:
                    action = noisyAction
                    action_eval = noisy_eval

        noisyAction = self.applyNoise(action)

        if __debug__ and bot.player.getSelected():
            print("")
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG" and self.getNoise() != 0:
                print("Evaluation of current state-action Q(s,a): ", round(self.critic.predict(state, noisyAction), 2))
            else:
                print("Evaluation of current state V(s): ", round(self.critic.predict(state), 2))
            print("Current action:\t", numpy.round(noisyAction, 2))

        return action, noisyAction

    def calculateTargetAndTDE(self, old_s, r, new_s, alive, a):
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            old_state_value = self.critic.predict(old_s, numpy.array([a]))
        else:
            old_state_value = self.critic.predict(old_s)

        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                updated_prediction = self.critic.predict_target_model(new_s, numpy.array([a]))
            else:
                updated_prediction = self.critic.predict_target_model(new_s)
            target += self.discount * updated_prediction
        td_error = target - old_state_value
        return target, td_error


    def train_critic_DPG(self, batch, get_evals = False):
        batch_len = len(batch[0])
        inputs_critic_states = numpy.zeros((batch_len, self.input_len))
        inputs_critic_actions = numpy.zeros((batch_len, self.action_len))
        targets_critic = numpy.zeros((batch_len, 1))
        idxs = batch[6] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else None
        importance_weights = batch[5] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else numpy.ones(batch_len)
        priorities = numpy.zeros_like(importance_weights)

        for sample_idx in range(batch_len):
            old_s, a, r, new_s = batch[0][sample_idx], batch[1][sample_idx], batch[2][sample_idx], batch[3][
                sample_idx]
            target = r
            if self.parameters.EXP_REPLAY_ENABLED:
                alive = new_s.size > 1
            else:
                alive = new_s is not None
            if alive:
                if self.parameters.DPG_USE_TARGET_MODELS:
                    estimationNewState = self.critic.predict_target_model(new_s, self.actor.predict_target_model(new_s))
                else:
                    estimationNewState = self.critic.predict(new_s, numpy.array([self.actor.predict(new_s)]))
                target += self.discount * estimationNewState
            estimationOldState = self.critic.predict(old_s, numpy.array([a]))
            td_e = target - estimationOldState
            if get_evals:
                priorities[sample_idx] = estimationOldState
            else:
                priorities[sample_idx] = td_e
            inputs_critic_states[sample_idx]  = old_s
            inputs_critic_actions[sample_idx] = a
            targets_critic[sample_idx] = target

        inputs_critic = [inputs_critic_states, inputs_critic_actions]
        self.critic.train(inputs_critic, targets_critic, importance_weights)

        return idxs, priorities


    def train_critic(self, batch, off_policy_weights):
        batch_len = len(batch[0])

        if self.parameters.CNN_REPR:
            inputShape = numpy.array([batch_len] + list(self.input_len[0]))
            inputs_critic = numpy.zeros(inputShape)
            if self.parameters.EXTRA_INPUT:
                extraInput = numpy.zeros((batch_len, self.input_len[1]))
                inputs_critic = [inputs_critic, extraInput]
        else:
            inputs_critic = numpy.zeros((batch_len, self.input_len))
        targets_critic = numpy.zeros((batch_len, 1))

        # Calculate input and target for critic
        idxs = batch[6] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else None
        importance_weights = batch[5] if self.parameters.PRIORITIZED_EXP_REPLAY_ENABLED else numpy.ones(batch_len)
        importance_weights *= off_policy_weights
        priorities = numpy.zeros_like(importance_weights)

        for sample_idx in range(batch_len):
            old_s, a, r, new_s = batch[0][sample_idx], batch[1][sample_idx], batch[2][sample_idx], batch[3][
                sample_idx]
            if self.parameters.EXP_REPLAY_ENABLED:
                alive = new_s.size > 1
            else:
                alive = new_s is not None
            target, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive, a)
            priorities[sample_idx] = td_e
            if self.parameters.CNN_REPR and self.parameters.EXTRA_INPUT:
                inputs_critic[0][sample_idx] = old_s[0]
                inputs_critic[1][sample_idx] = old_s[1]
            else:
                inputs_critic[sample_idx] = old_s
            targets_critic[sample_idx] = target

        # Train:
        self.critic.train(inputs_critic, targets_critic, importance_weights)

        return idxs, priorities



    def calculateTarget_Actor(self, old_s, a, td_e):
        target = None
        if self.acType == "CACLA" or self.acType == "DPG":
            if td_e > 0:
                mu_s = self.actor.predict(old_s)
                target = mu_s + (a - mu_s)
            elif td_e < 0 and self.parameters.CACLA_UPDATE_ON_NEGATIVE_TD:
                mu_s = self.actor.predict(old_s)
                target = mu_s - (a - mu_s)
        elif self.acType == "Standard":
            mu_s = self.actor.predict(old_s)
            target = mu_s + td_e * (a - mu_s)

        return target


    def load(self, modelName):
        if modelName is not None:
            path = modelName
            self.critic.load(path)
            self.actor.load(path)

    def save(self, path, name = ""):
        self.actor.save(path, name)
        self.critic.save(path, name)


    def setNoise(self, val):
        self.std = val

    def setTemperature(self, val):
        self.temperature = val

    def getTemperature(self):
        return None

    def reset(self):
        self.latestTDerror = None
        self.ornUhlPrev = numpy.zeros(self.action_len)


    def resetQValueList(self):
        self.qValues = []

    def getNoise(self):
        return self.std

    def getTDError(self):
        return self.latestTDerror

    def getQValues(self):
        return self.qValues