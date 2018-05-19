import numpy
import heapq

class QLearn(object):
    def __repr__(self):
        return "Q-learning"

    def __init__(self, numOfNNbots, numOfHumans, network, parameters):
        self.num_NNbots = numOfNNbots
        self.num_humans = numOfHumans
        self.network = network
        self.temporalDifference = parameters.TD
        self.parameters = parameters
        self.latestTDerror = None
        self.qValues = []
        self.input_len = parameters.STATE_REPR_LEN
        self.output_len = network.num_actions
        self.discrete = True
        self.epsilon = parameters.EPSILON

    def load(self, modelName):
        if modelName is not None:
            self.network.load(modelName)

    def reset(self):
        self.latestTDerror = None
        if self.parameters.NEURON_TYPE == "LSTM":
            self.network.reset_hidden_states()

    def reset_weights(self):
        self.network.reset_weights()

    def resetQValueList(self):
        self.qValues = []

    def testNetwork(self, bot, newState):
        self.network.setEpsilon(0)
        player = bot.getPlayer()
        return self.decideMove(newState, player)

    def calculateTargetForAction(self, newState, reward, alive):
        target = reward
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            action_Q_values = self.network.predict_target_network(newState)
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.network.getDiscount() * action_Q_values[newActionIdx]
        return target

    def calculateTarget(self, old_s, a, r, new_s):
        old_q_values = self.network.predict(old_s)
        alive = (new_s is not None)
        updated_action_value = self.calculateTargetForAction(new_s, r, alive)
        old_q_values[a] = updated_action_value
        return old_q_values

    def calculateTDError(self, experience):
        old_s, a, r, new_s, _ = experience
        alive = new_s is not None
        state_Q_values = self.network.predict(old_s)
        target = self.calculateTargetForAction(new_s, r, alive)
        q_value_of_action = state_Q_values[a]
        td_error = target - q_value_of_action
        return td_error

    def calculateTDError_ExpRep_Lstm(self, exp):
        old_s, a, r, new_s, _ = exp
        alive = new_s is not None
        state_Q_values = self.network.predict_single_trace_LSTM(old_s, False)
        target = r
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            action_Q_values = self.network.predict_single_trace_LSTM(new_s, False)

            # TODO: Maybe this is not so smart at all! The hidden state of this prediction will be passed on, so it might lead to weird hidden states if we train twice on one?

            newActionIdx = numpy.argmax(action_Q_values)
            target += self.network.getDiscount() * action_Q_values[newActionIdx]
        q_value_of_action = state_Q_values[a]
        td_error = target - q_value_of_action
        return td_error

    def train(self, batch):
        batch_len = len(batch)
        inputs = numpy.zeros((batch_len, self.input_len))
        targets = numpy.zeros((batch_len, self.output_len))
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s, _ = sample
            # No new state: dead
            inputs[sample_idx] = old_s
            targets[sample_idx] = self.calculateTarget(old_s, a, r, new_s)
        self.network.trainOnBatch(inputs, targets)

    def train_LSTM_batch(self, batch):
        len_batch = len(batch)
        len_trace = self.parameters.MEMORY_TRACE_LEN
        old_states = numpy.zeros((len_batch, len_trace, self.input_len))
        new_states = numpy.zeros((len_batch, len_trace, self.input_len))

        action = numpy.zeros((len_batch, len_trace))
        reward = numpy.zeros((len_batch, len_trace))

        for i in range(len_batch):
            for j in range(len_trace):
                old_states[i, j, :] = batch[i][j][0]
                action[i, j] = batch[i][j][1]
                reward[i, j] = batch[i][j][2]
                new_states[i, j, :] = batch[i][j][3]

        #TODO: Why is len_batch needed? The shape seems to be 32, 10, 407 for some reason
        #print("Shape old_states: ", numpy.shape(old_states))
        target = self.network.predict(old_states, len_batch)

        #TODO: agent might have died so do not predict reward there with new State as new state is None then
        target_val = self.network.predict_target_network(new_states, len_batch)

        for i in range(len_batch):
            for j in range(self.parameters.TRACE_MIN, len_trace):
                a = numpy.argmax(target_val[i][j])
                # TODO: target is reward if the agent died.
                target[i][j][int(action[i][j])] = reward[i][j] + self.parameters.DISCOUNT * (target_val[i][j][a])
        #print("Weights before training:")
        #print(self.network.valueNetwork.get_weights()[0])
        self.network.trainOnBatch(old_states, target)
        #print("After training:")
        #print(self.network.valueNetwork.get_weights()[0])


#TODO: With lstm neurons we need one network python object per learning agent in the environment.
#TODO: !! important if at some point we want to train multiple lstm agents

    def train_LSTM(self, batch):
        if self.parameters.EXP_REPLAY_ENABLED:
            # The last memory is not a memory trace, therefore useless for exp replay lstm
            if len(batch) > self.parameters.MEMORY_BATCH_LEN:
                self.train_LSTM_batch(batch[:-1])
        else:
            old_s, a, r, new_s, reset = batch[-1]

            target = self.network.predict(old_s)
            # Debug: print predicted q values:
            average_value = round(numpy.mean(target), 2)
            q_value = round(target[a], 2)
            print("Expected Q-value: ", average_value, " Q(s,a) of current action: ", q_value)

            # Calculate target for action that we took:
            alive = (new_s is not None)
            target_for_action = self.calculateTargetForAction(new_s, r, alive)
            self.latestTDerror = target_for_action - target[a] # BookKeeping
            target[a] = target_for_action
            print("Reward: ", round(r, 2))
            print("Target for action: ", round(target_for_action, 2))
            print("TD-Error: ", self.latestTDerror)
            loss = self.network.trainOnBatch(old_s, target)
            print("Loss: ", loss)
            print("")

#TODO: Get loss from all trainOnBatch calls, store them in an array and plot them


    def learn(self, batch):
        if self.parameters.NEURON_TYPE == "LSTM":
            self.train_LSTM(batch)
        else:
            self.train(batch)

        #Book keeping. batch[-1] is the current experience:
        if not self.parameters.NEURON_TYPE == "LSTM":
            currentExp = batch[-1]
            if self.parameters.NEURON_TYPE == "MLP" or not self.parameters.EXP_REPLAY_ENABLED:
                self.latestTDerror = self.calculateTDError(currentExp)
            else:
                self.latestTDerror = self.calculateTDError_ExpRep_Lstm(currentExp)

    def getNoiseLevel(self):
        return self.epsilon

    def updateNoise(self):
        self.epsilon *= self.parameters.NOISE_DECAY

    def updateNetworks(self, time):
        self.updateTargetModel(time)
        self.updateActionModel(time)

    def updateTargetModel(self, time):
        if time % self.parameters.TARGET_NETWORK_MAX_STEPS == 0:
           self.network.updateTargetNetwork()

    def updateActionModel(self, time):
        if self.parameters.NEURON_TYPE == "LSTM" and time % self.parameters.UPDATE_LSTM_MOVE_NETWORK == 0:
            self.network.updateActionNetwork()


    def decideMove(self, newState, bot):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.epsilon:
            newActionIdx = numpy.random.randint(len(self.network.getActions()))
            self.qValues.append(float("NaN"))
            explore = True
            if __debug__:
                print("Explore")
                bot.setExploring(True)
        else:
            explore = False
            if bot.parameters.NEURON_TYPE == "MLP":
                # Take action based on greediness towards Q values
                q_Values = self.network.predict(newState)
            else:
                q_Values = self.network.predict_action_network(newState)
            newActionIdx = numpy.argmax(q_Values)
            # Book keeping:
            self.qValues.append(q_Values[newActionIdx])
            if __debug__:
                bot.setExploring(False)

        if __debug__  and not explore:
            average_value = round(numpy.mean(q_Values), 1)
            q_value = round(q_Values[newActionIdx], 1)
            print("Expected Q-value: ", average_value, " Q(s,a) of current action: ", q_value)
            print("")
        newAction = self.network.actions[newActionIdx]

        return newActionIdx, newAction

    def save(self, path):
        self.network.saveModel(path)

    def setNoise(self, val):
        self.epsilon = 0

    def getNetwork(self):
        return self.network

    def getQValues(self):
        return self.qValues

    def getTDError(self):
        return self.latestTDerror