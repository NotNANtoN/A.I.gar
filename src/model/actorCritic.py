import keras
import numpy
import math
import tensorflow as tf
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

        if modelName is not None:
            self.load(modelName)
            return

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

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(loss='mse', optimizer=optimizer)
        self.target_model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "/value_model.h5")
            self.target_model = load_model(path + "/value_model.h5")

    def predict(self, state):
        return self.model.predict(numpy.array([state]))[0][0]

    def predict_target_model(self, state):
        return self.target_model.predict(numpy.array([state]))[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + "value_model.h5")


class PolicyNetwork(object):
    def __init__(self, parameters, discrete, modelName=None):
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

        self.discrete = discrete

        if discrete:
            self.actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
                            eject in [0, 1]]
            # Filter out actions that do a split and eject at the same time
            # Filter eject and split actions for now
            for action in self.actions[:]:
                if action[2] or action[3]:
                    self.actions.remove(action)
            self.num_actions = len(self.actions)
            self.num_outputs = self.num_actions
        else:
            self.num_outputs = 2  #x, y, split, eject all continuous between 0 and 1
            if self.parameters.ENABLE_SPLIT == True:
                self.num_outputs += 1
            if self.parameters.ENABLE_EJECT == True:
                self.num_outputs += 1

        if modelName is not None:
            self.load(modelName)
            return

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + self.num_outputs))
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
                    Dense(self.num_outputs, activation=relu_max, bias_initializer=initializer
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
            if discrete:
                self.model.add(
                    Dense(self.num_outputs, activation="softmax", bias_initializer=initializer
                          , kernel_initializer=initializer))
            else:
                self.model.add(
                    Dense(self.num_outputs, activation=relu_max, bias_initializer=initializer
                          , kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)
        self.model.compile(loss='mse', optimizer=optimizer)

    def load(self, modelName=None):
        if modelName is not None:
            path = modelName
            self.loadedModelName = modelName
            self.model = load_model(path + "/actor_model.h5", custom_objects={"relu_max": relu_max})

    def predict(self, state):
        return self.model.predict(numpy.array([state]))[0]

    def train(self, inputs, targets):
        if self.parameters.ACTOR_REPLAY_ENABLED:
            self.model.train_on_batch(inputs, targets)
        else:
            self.model.train_on_batch(numpy.array([inputs]), numpy.array([targets]))

    def save(self, path):
        self.model.save(path + "actor" + "_model.h5")

    def getTarget(self, action, state):
        if self.discrete:
            # todo: this does not work yet. we want to slightly increase the p(s,a), not set it to 1
            target = numpy.zeros(self.num_actions)
            # target = self.model.predict(state)[0]
            target[action] = 1
        else:
            target = action
        return numpy.array([target])

    def getAction(self, action_idx):
        return self.actions[action_idx]


class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters, num_bots, discrete, modelName=None):
        self.acType = parameters.ACTOR_CRITIC_TYPE
        self.num_bots = num_bots
        self.actor = PolicyNetwork(parameters, discrete, modelName)
        self.critic = ValueNetwork(parameters, modelName)
        self.parameters = parameters
        self.std = self.parameters.GAUSSIAN_NOISE
        self.noise_decay_factor = self.parameters.NOISE_DECAY
        self.discrete = discrete
        self.steps = 0
        self.input_len = parameters.STATE_REPR_LEN
        # Bookkeeping:
        self.latestTDerror = None
        self.qValues = []

    def updateNoise(self):
        self.std *= self.noise_decay_factor

    def updateCriticNetworks(self, time):
        if time % self.parameters.TARGET_NETWORK_STEPS == 0:
            self.critic.update_target_model()

    def updateNetworks(self, time):
        self.updateCriticNetworks(time)

    def learn(self, batch):
        self.train_critic(batch)
        if self.parameters.ACTOR_REPLAY_ENABLED:
            self.train_actor_batch(batch)
        else:
            currentExp = batch[-1]
            self.train_actor(currentExp)

    def decideMove(self, state, bot):
        actions = self.actor.predict(state)
        std_dev = self.std
        apply_normal_dist = [numpy.random.normal(output, std_dev) for output in actions]
        clipped = numpy.clip(apply_normal_dist, 0, 1)
        if self.discrete:
            action_idx = numpy.argmax(clipped)
            action = self.actor.getAction(action_idx)
        else:
            action_idx = None
            action = clipped

        if __debug__:
            print("")
            print("V(s): ", round(self.critic.predict(state), 2))
            print("Current action:\t", numpy.round(action, 2))
            print("")

        return action_idx, action

    def calculateTargetAndTDE(self, old_s, r, new_s, alive):
        old_state_value = self.critic.predict(old_s)
        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            updated_prediction = self.critic.predict_target_model(new_s)
            target += self.parameters.DISCOUNT * updated_prediction
        td_error = target - old_state_value
        return target, td_error

    def train_critic(self, batch):
        len_batch = len(batch)
        inputs_critic = numpy.zeros((len_batch, self.input_len))
        targets_critic = numpy.zeros((len_batch, 1))
        # Calculate input and target for critic
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            target, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            inputs_critic[sample_idx] = old_s
            targets_critic[sample_idx] = target
        # Debug info:
        if __debug__:
            old_s, a, r, new_s, _ = batch[-1]
            alive = new_s is not None
            target, self.latestTDerror = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            self.qValues.append(target)
        self.critic.train(inputs_critic, targets_critic)

#TODO: check critic training, the resulting values are weirdly negative for mass 200


    def train_actor(self, currentExp):
        old_s, a, r, new_s, _ = currentExp
        _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, new_s is not None)
        target = self.calculateTarget_Actor(old_s, a, td_e)
        if target is not None:
            self.actor.train(old_s, target)

    def calculateTarget_Actor(self, old_s, a, td_e):
        target = None
        if self.acType == "CACLA":
            if td_e > 0:
                mu_s = self.actor.predict(old_s)
                target = mu_s + (a - mu_s) #/ self.std ** 2
        elif self.acType == "Standard":
            mu_s = self.actor.predict(old_s)
            target = mu_s + td_e * (a - mu_s) #/ self.std ** 2
        # TODO: Do we need to clip targets to 0-1 range??? How does backprop work if it wants to tune the network to attain unachievable results?

        return target

    def train_actor_batch(self, batch):
        len_batch = len(batch)
        len_output = self.actor.num_outputs
        inputs = numpy.zeros((len_batch, self.input_len))
        targets = numpy.zeros((len_batch, len_output))

        # Calculate input and target for actor
        count = 0
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive)
            target = self.calculateTarget_Actor(old_s, a, td_e)
            if target is not None:
                inputs[count] = old_s
                targets[count] = target
                count += 1
        if count > 0:
            inputs = inputs[:count]
            targets = targets[:count]
            if __debug__:
                if batch[-1][0] is inputs[-1]:
                    print("Target for current experience:", numpy.round(targets[-1], 2))
                print("Last predicted action:\t", numpy.round(self.actor.predict(inputs[-1]), 2))
                print("Last Target:\t", numpy.round(targets[-1], 2))
                print("Actor trained on number of samples: ", count)
            self.actor.train(inputs, targets)
            if __debug__:
                print("Predicted action after training:\t", numpy.round(self.actor.predict(inputs[-1]), 2))

    # This is deprecated:
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
            std_dev = self.std
            actor_action = self.actor.predict(old_s)
            gradient_of_log_prob_target_actor = actor_action + (a - actor_action) / (std_dev * std_dev)
            gradient = self.actor.getGradient(old_s, gradient_of_log_prob_target_actor)
            single_weight_change_actor = gradient * td_error
            total_weight_changes_actor += single_weight_change_actor

        self.critic.train(inputs_critic, targets_critic)
        self.actor.train(total_weight_changes_actor)

    def load(self, modelName):
        if modelName is not None:
            path = modelName
            self.critic.load(path)
            self.actor.load(path)

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def setNoise(self, val):
        self.std = val

    def setTemperature(self, val):
        self.temperature = val

    def reset(self):
        self.latestTDerror = None

    def resetQValueList(self):
        self.qValues = []

    def getNoiseLevel(self):
        return self.std

    def getTDError(self):
        return self.latestTDerror

    def getQValues(self):
        return self.qValues
