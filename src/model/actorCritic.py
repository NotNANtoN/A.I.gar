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
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN_POLICY
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT_POLICY

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1_POLICY
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2_POLICY
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3_POLICY

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
        return self.model.predict(numpy.array([state]))[0]
        

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
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT_POLICY

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
                Dense(num_outputs, activation=relu_max, bias_initializer=initializer
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
        return self.model.predict(numpy.array([state]))

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path):
        self.model.save(path + "actor"+ "_model.h5")

class Actor(object):
    def __init__(self, parameters):
        self.policyNetwork = PolicyNetwork(parameters)

    def predict(self, state):
        return self.policyNetwork.predict(state)

    # Makes the given target outputs more likely
    def train_CACLA(self, states, targets):
        self.policyNetwork.train()

    def train(self, weight_changes):
        weights = self.policyNetwork.get_weights()
        self.policyNetwork.set_weights(weights + weight_changes)



    def getGradient(self, input, output):
        return self.policyNetwork.get_gradient(input, output)

class Critic(object):
    def __init__(self, parameters):
        self.valueNetwork = ValueNetwork(parameters)

    def predict(self, state):
        return self.valueNetwork.predict(state)

    def train(self, inputs, outputs):
        self.valueNetwork.train_on_batch(inputs, outputs)

class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters):
        self.actor = PolicyNetwork(parameters)
        self.critic = ValueNetwork(parameters)
        self.parameters = parameters
        self.parameters.std_dev = self.parameters.EPSILON
        self.discrete = False

    def learn(self, batch):
        self.train_CACLA(batch)

    def decideMove(self, state):
        return self.actor.predict(state)[0]

    def train_CACLA(self, batch):
        inputs_critic = []
        targets_critic = []
        inputs_actor = []
        targets_actor = []
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

            if td_error > 0:
                inputs_actor.append(old_s)
                targets_actor.append(a)

        self.actor.train(inputs_actor, targets_actor)
        self.critic.train(inputs_critic, targets_critic)


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

    def act_train(self, state):
        actions = self.actor.predict(state)
        std_dev = self.parameters.std_dev
        actions = [numpy.random.normal(mean, std_dev) for mean in actions]
        return actions

    def reset(self):
        pass



