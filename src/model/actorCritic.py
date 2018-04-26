import keras
import numpy
import math
import tensorflow as tf
import importlib
from keras.layers import Dense, LSTM, Softmax
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model
from keras import backend as K

def relu_max(x):
    return K.relu(x, max_value=1)


class ValueNetwork(object):
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        self.gpus = self.parameters.GPUS

        self.learningRate = self.parameters.ALPHA_POLICY
        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3

        num_outputs = 1  # value for state

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + num_outputs))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.model = Sequential()
                self.model.add(
                    Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                          bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                self.model.add(
                    Dense(num_outputs, activation='linear', bias_initializer=initializer
                          , kernel_initializer=initializer))
                self.model = multi_gpu_model(self.model, gpus=self.gpus)
        else:
            self.model = Sequential()
            hidden1 = None
            if self.parameters.NEURON_TYPE == "MLP":
                hidden1 = Dense(self.hiddenLayer1, input_dim=self.stateReprLen,
                                activation=self.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)
            elif self.parameters.NEURON_TYPE == "LSTM":
                hidden1 = LSTM(self.hiddenLayer1, input_shape=(self.stateReprLen, 1),
                               activation=self.activationFuncHidden,
                               bias_initializer=initializer, kernel_initializer=initializer)

            self.model.add(hidden1)
            # self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden2)
                # self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden3)
                # self.valueNetwork.add(Dropout(0.5))

            self.model.add(
                Dense(num_outputs, activation='linear', bias_initializer=initializer
                      , kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)
        self.model.compile(loss='mse', optimizer=optimizer)

        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.loadedModelName = modelName
            self.model = load_model(path + "/value_model.h5")

    def predict(self, state):
        return self.model.predict(state)[0]

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path):
        self.model.save(path + "value" + "_model.h5")


class PolicyNetwork(object):
    def __init__(self, parameters, modelName = None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        self.gpus = self.parameters.GPUS

        self.learningRate = self.parameters.ALPHA_POLICY
        self.optimizer = self.parameters.OPTIMIZER_POLICY
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY
        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1_POLICY
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2_POLICY
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3_POLICY

        num_outputs = 4 # x, y, split, eject all continuous between 0 and 1

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + num_outputs))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.model = Sequential()
                self.model.add(
                    Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                          bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.model.add(
                        Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                self.model.add(
                    Dense(num_outputs, activation=relu_max, bias_initializer=initializer
                          , kernel_initializer=initializer))
                self.model = multi_gpu_model(self.model, gpus=self.gpus)
        else:
            self.model = Sequential()
            hidden1 = None
            if self.parameters.NEURON_TYPE == "MLP":
                hidden1 = Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)
            elif self.parameters.NEURON_TYPE == "LSTM":
                hidden1 = LSTM(self.hiddenLayer1, input_shape=(self.stateReprLen, 1),
                               activation=self.activationFuncHidden,
                               bias_initializer=initializer, kernel_initializer=initializer)

            self.model.add(hidden1)
            # self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden2)
                # self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.model.add(hidden3)
                # self.valueNetwork.add(Dropout(0.5))

            self.model.add(
                Dense(num_outputs, activation="sigmoid", bias_initializer=initializer
                      , kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)
        self.model.compile(loss='mse', optimizer=optimizer)

        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.loadedModelName = modelName
            self.model = load_model(path + "/actor_model.h5")

    def predict(self, state):
        return self.model.predict(state)[0]

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path):
        self.model.save(path + "actor"+ "_model.h5")

    def load(self, path):
        self.model = load_model(path)

class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters):
        self.actor = PolicyNetwork(parameters)
        self.critic = ValueNetwork(parameters)
        self.parameters = parameters
        self.parameters.std_dev = self.parameters.EPSILON
        self.discrete = False
        self.steps = 0
        self.input_len = parameters.STATE_REPR_LEN
        # Bookkeeping:
        self.lastTDE = None
        self.qValues = []


    def learn(self, batch):
        self.steps += 1
        self.train_CACLA(batch)
        if self.steps % self.parameters.TARGET_NETWORK_MAX_STEPS == 0:
            pass
            # TODO: update target value network. implement a target network for that
            # TODO: take care that the number of steps reflects number of frames, and not number of frames * number of bots that use actor critic

    #def decideMove(self, state):
    #    return self.actor.predict(state)[0]

    def decideMove(self, state):
        actions = self.actor.predict(state)
        std_dev = self.parameters.std_dev
        apply_normal_dist = [numpy.random.normal(output, std_dev) for output in actions]
        clipped = numpy.clip(apply_normal_dist, 0, 1)

        return clipped

    def calculateTargetAndTDE(self, old_s, r, new_s, alive):
        old_state_value = self.critic.predict(old_s)
        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            updated_prediction = self.critic.predict(new_s)
            target += self.parameters.DISCOUNT * updated_prediction
        td_error = target - old_state_value
        return target, td_error

    def train_critic_CACLA(self, batch):
        len_batch = len(batch)
        inputs_critic = numpy.zeros((len_batch, self.input_len))
        targets_critic = numpy.zeros((len_batch, 1))

        # Calculate input and target for critic
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s = sample
            alive = new_s is not None
            target, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            inputs_critic[sample_idx] = old_s
            targets_critic[sample_idx] = target
        old_s, a, r, new_s =  batch[-1]
        alive = new_s is not None
        target, self.lastTDE = self.calculateTargetAndTDE(old_s, r, new_s, alive)
        self.qValues.append(target)
        self.critic.train(inputs_critic, targets_critic)

    def train_actor_CACLA(self, currentExp):
        old_s, a, r, new_s = currentExp
        a = numpy.array([a])
        _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, new_s is not None)
        if td_e > 0:
            input_actor = old_s
            target_actor = a
            self.actor.train(input_actor, target_actor)

    def train_actor_batch_CACLA(self, batch):
        len_batch = len(batch)
        inputs = numpy.zeros((len_batch, self.input_len))
        targets = numpy.zeros((len_batch, 4))

        # Calculate input and target for actor
        count = 0
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s = sample
            a = numpy.array([a])
            alive = new_s is not None
            _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            if td_e > 0:
                inputs[sample_idx] = old_s
                targets[sample_idx] = a
                count += 1
        if count > 0:
            inputs = inputs[0:count]
            targets = targets[0:count]
            self.actor.train(inputs, targets)

    def train_CACLA(self, batch):
        # TODO: actor should not be included in the replays.. I think???
        # TODO: add target network for value net
        self.train_critic_CACLA(batch)
        currentExp = batch[-1]
        self.train_actor_CACLA(currentExp)
        #self.train_actor_batch_CACLA(batch)



    def train(self, batch):
        inputs_critic = []
        targets_critic = []
        total_weight_changes_actor = 0
        for sample in batch:
            # Calculate input and target for critic
            old_s, a, r, new_s = sample
            alive = new_s is not None
            old_state_value = self.critic.predict(old_s)
            target = r
            if alive:
                # The target is the reward plus the discounted prediction of the value network
                updated_prediction = self.critic.predict(new_s)
                target += self.parameters.discount * updated_prediction
            td_error = target - old_state_value
            inputs_critic.append(old_s)
            targets_critic.append(target)

            # Calculate weight change of actor:
            std_dev = self.parameters.std_dev
            actor_action = self.actor.predict(old_s)
            gradient_of_log_prob_target_actor = actor_action + (a - actor_action) / (std_dev * std_dev)
            gradient = self.actor.getGradient(old_s, gradient_of_log_prob_target_actor)
            single_weight_change_actor = gradient * td_error
            total_weight_changes_actor += single_weight_change_actor

        self.critic.train(inputs_critic, targets_critic)
        self.actor.train(total_weight_changes_actor)

    def load(self, modelName):
        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.critic.load(path + "/critic_model.h5")
            self.actor.load(path + "/actor_model.h5")

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def reset(self):
        pass

    def getTDError(self):
        return self.lastTDE

    def getQValues(self):
        return self.qValues

