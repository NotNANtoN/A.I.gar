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


#TODO: put ornstein uhlenbeck into usable function
class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ValueNetwork(object):
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN

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

        if self.parameters.INITIALIZER == "glorot_uniform":
            initializer = keras.initializers.glorot_uniform()
        elif self.parameters.INITIALIZER == "glorot_normal":
            initializer = keras.initializers.glorot_normal()
        else:
            weight_initializer_range = math.sqrt(6 / (self.stateReprLen + 1))
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                           maxval=weight_initializer_range, seed=None)


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

        self.model.add(Dense(num_outputs, activation='linear', bias_initializer=initializer, kernel_initializer=initializer))

        optimizer = keras.optimizers.Adam(lr=self.learningRate)

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
        return self.model.predict(state)[0][0]

    def predict_target_model(self, state):
        return self.target_model.predict(state)[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "value_model.h5")


class PolicyNetwork(object):
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.learningRate = self.parameters.DPG_ACTOR_ALPHA
            self.layers = parameters.DPG_ACTOR_LAYERS
        else:
            self.learningRate = self.parameters.ALPHA_POLICY
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
            initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range, maxval=weight_initializer_range, seed=None)

        inputState = keras.layers.Input((self.stateReprLen,))
        previousLayer = inputState
        for neuronNumber in self.layers:
            previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden, bias_initializer=initializer,
                                  kernel_initializer=initializer)(previousLayer)

        output = Dense(self.num_outputs, activation="sigmoid", bias_initializer=initializer, kernel_initializer=initializer)(
            previousLayer)
        self.model = keras.models.Model(inputs=inputState, outputs=output)

        optimizer = keras.optimizers.Adam(lr=self.learningRate)

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
        return self.model.predict(state)[0]

    def predict_target_model(self, state):
        return self.target_model.predict(state)


    def train(self, inputs, targets):
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
    def __init__(self, parameters, modelName=None):
        self.parameters = parameters
        self.loadedModelName = None
        self.stateReprLen = self.parameters.STATE_REPR_LEN
        self.learningRate = self.parameters.DPG_CRITIC_ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        self.activationFuncHidden = self.parameters.DPG_CRITIC_FUNC
        layers = self.parameters.DPG_CRITIC_LAYERS

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
        inputState = keras.layers.Input((self.stateReprLen,))
        inputAction = keras.layers.Input((self.num_actions_inputs,))
        previousLayer = inputState
        for idx, neuronNumber in enumerate(layers):
            if idx == parameters.DPG_FEED_ACTION_IN_LAYER - 1:
                mergeLayer = keras.layers.concatenate([previousLayer, inputAction])
                previousLayer = mergeLayer
            previousLayer = Dense(neuronNumber, activation=self.activationFuncHidden, bias_initializer=initializer,
                                  kernel_initializer=initializer, kernel_regularizer=regularizer)(previousLayer)


        output = Dense(1, activation="linear", bias_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=regularizer)(previousLayer)
        self.model = keras.models.Model(inputs=[inputState, inputAction], outputs=output)


        optimizer = keras.optimizers.Adam(lr=self.learningRate)

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
        return self.model.predict([state, action])[0][0]

    def predict_target_model(self, state, action):
        return self.target_model.predict([state, action])[0][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def softlyUpdateTargetModel(self):
        tau = self.parameters.DPG_TAU
        targetWeights = self.target_model.get_weights()
        modelWeights = self.model.get_weights()
        newWeights = [targetWeights[idx] * (1 - tau) + modelWeights[idx] * tau for idx in range(len(modelWeights))]
        self.target_model.set_weights(newWeights)

    def train(self, inputs, targets):
        self.model.train_on_batch(inputs, targets)

    def save(self, path, name):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save(path + name + "actionValue_model.h5")


class ActorCritic(object):
    def __repr__(self):
        return "AC"

    def __init__(self, parameters):
        self.discrete = False
        self.acType = parameters.ACTOR_CRITIC_TYPE
        self.parameters = parameters
        self.std = self.parameters.GAUSSIAN_NOISE
        self.noise_decay_factor = self.parameters.NOISE_DECAY
        self.steps = 0
        self.input_len = parameters.STATE_REPR_LEN
        self.action_len = 2 + self.parameters.ENABLE_SPLIT + self.parameters.ENABLE_EJECT


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
        combinedModel.compile(optimizer=keras.optimizers.Adam(lr=actor.learningRate), loss="mse")
        return combinedModel

    def initializeNetwork(self, loadPath, networks=None):
        if networks is None or networks == {}:
            if networks is None:
                networks = {}
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                self.actor = PolicyNetwork(self.parameters, loadPath)
                self.critic = ActionValueNetwork(self.parameters, loadPath)
                self.combinedActorCritic = self.createCombinedActorCritic(self.actor, self.critic)
                networks["MU(S)"] = self.actor
                networks["Q(S,A)"] = self.critic
                networks["Actor-Critic-Combo"] = self.combinedActorCritic
            else:
                self.actor = PolicyNetwork(self.parameters, loadPath)
                self.critic = ValueNetwork(self.parameters, loadPath)
                networks["MU(S)"] = self.actor
                networks["V(S)"] = self.critic
        else:
            self.actor  = networks["MU(S)"]
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
                self.critic = networks["Q(S,A)"]
                self.combinedActorCritic = networks["Actor-Critic-Combo"]
            else:
                self.critic = networks["V(S)"]
        for network in networks:
            print(network + " summary:")
            if network == "Actor-Critic-Combo":
                networks[network].summary()
                continue
            networks[network].model.summary()
        return networks

    def updateNoise(self):
        self.std *= self.noise_decay_factor

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


    def learn(self, batch):
        if self.parameters.ACTOR_CRITIC_TYPE == "DPG":
            self.train_critic_DPG(batch)
            if self.parameters.DPG_USE_DPG_ACTOR_TRAINING:
                self.train_actor_DPG(batch)
            if self.parameters.DPG_USE_CACLA:
                self.train_actor_batch(batch)
        else:
            self.train_critic(batch)
            self.train_actor_batch(batch)

    def train_actor_DPG(self, batch):
        len_batch = len(batch)
        inputs = numpy.zeros((len_batch, self.parameters.STATE_REPR_LEN))
        targets = numpy.zeros((len_batch, 1))

        if self.parameters.DPG_USE_CACLA:
            targets_CACLA = numpy.zeros((len_batch, self.actor.num_outputs))



        # Calculate input and target for actor
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            oldPrediction = self.combinedActorCritic.predict(old_s)[0]
            inputs[sample_idx] = old_s
            targets[sample_idx] = oldPrediction + self.parameters.DPG_Q_VAL_INCREASE

        actions = self.actor.predict(batch[-1][0])
        loss = self.combinedActorCritic.train_on_batch(inputs, targets)
        actionsAfter = self.actor.predict(batch[-1][0])
        #print("Before:", numpy.round(actions,2), " After:", numpy.round(actionsAfter,2))



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
            _, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive, a)
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

            actions = self.actor.predict(batch[-1][0])
            self.actor.train(inputs, targets)
            actionsAfter = self.actor.predict(batch[-1][0])
            #print("Before:", numpy.round(actions, 2), " After:", numpy.round(actionsAfter, 2))


            if __debug__:
                print("Predicted action after training:\t", numpy.round(self.actor.predict(inputs[-1]), 2))

    def applyNoise(self, action):
        #Gaussian Noise:
        apply_normal_dist = [numpy.random.normal(output, self.std) for output in action]
        return numpy.clip(apply_normal_dist, 0, 1)
        #TODO: add Ornstein-Uhlenbeck process noise with theta=0.15 and sigma=0.2


    def decideMove(self, state, bot):
        action = self.actor.predict(state)
        noisyAction = self.applyNoise(action)

        if __debug__ and bot.player.getSelected():
            print("")
            if self.parameters.ACTOR_CRITIC_TYPE == "DPG" and self.getNoise() != 0:
                print("Evaluation of current state-action Q(s,a): ", round(self.critic.predict(state, noisyAction), 2))
            else:
                print("Evaluation of current state V(s): ", round(self.critic.predict(state), 2))
            print("Current action:\t", numpy.round(noisyAction, 2))

        return None, noisyAction

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
            target += self.parameters.DISCOUNT * updated_prediction
        td_error = target - old_state_value
        return target, td_error


    def train_critic_DPG(self, batch):
        target, td_e = None, None
        len_batch = len(batch)
        inputs_critic_states = numpy.zeros((len_batch, self.input_len))
        inputs_critic_actions = numpy.zeros((len_batch, self.action_len))
        targets_critic = numpy.zeros((len_batch, 1))

        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            target = r
            if alive:
                if self.parameters.DPG_USE_TARGET_MODELS:
                    estimationNewState = self.critic.predict_target_model(new_s, self.actor.predict_target_model(new_s))
                else:
                    estimationNewState = self.critic.predict(new_s, numpy.array([self.actor.predict(new_s)]))
                target += self.parameters.DISCOUNT * estimationNewState
            inputs_critic_states[sample_idx]  = old_s
            inputs_critic_actions[sample_idx] = a
            targets_critic[sample_idx] = target
        inputs_critic = [inputs_critic_states, inputs_critic_actions]
        self.qValues.append(target)
        self.critic.train(inputs_critic, targets_critic)


    def train_critic(self, batch):
        target, td_e = None, None
        len_batch = len(batch)
        inputs_critic = numpy.zeros((len_batch, self.input_len))
        targets_critic = numpy.zeros((len_batch, 1))
        # Calculate input and target for critic
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            alive = new_s is not None
            target, td_e = self.calculateTargetAndTDE(old_s, r, new_s, alive, a)
            inputs_critic[sample_idx] = old_s
            targets_critic[sample_idx] = target

        # Debug info:
        if target and td_e:
            self.qValues.append(target)
            self.latestTDerror = td_e

        # Train:
        self.critic.train(inputs_critic, targets_critic)


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

    def resetQValueList(self):
        self.qValues = []

    def getNoise(self):
        return self.std

    def getTDError(self):
        return self.latestTDerror

    def getQValues(self):
        return self.qValues
