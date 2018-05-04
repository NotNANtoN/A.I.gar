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
        self.time = 0
        self.input_len = parameters.STATE_REPR_LEN
        self.output_len = network.num_actions
        self.discrete = True

    def load(self, modelName):
        if modelName is not None:
            self.network.load(modelName)

    def reset(self):
        self.latestTDerror = None
        if self.parameters.NEURON_TYPE == "LSTM":
            self.network.reset_hidden_states_prediction()

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

        self.network.trainOnBatch(old_states, target)

    def train_LSTM(self, batch):
        if self.parameters.EXP_REPLAY_ENABLED:
            # The last memory is not a memory trace, therefore useless for exp replay lstm
            if len(batch) > self.parameters.MEMORY_BATCH_LEN:
                self.train_LSTM_batch(batch[:-1])
        else:
            old_s, a, r, new_s, reset = batch[-1]
            target = self.calculateTarget(old_s, a, r, new_s)
            self.network.trainOnBatch(old_s, target)


    def learn(self, batch):
        self.time += 1 * self.parameters.FRAME_SKIP_RATE

        if self.parameters.NEURON_TYPE == "LSTM":
            self.train_LSTM(batch)
        else:
            self.train(batch)

        #Book keeping. batch[-1] is the current experience:
        if __debug__ and self.parameters.NEURON_TYPE == "MLP":
            currentExp = batch[-1]
            self.latestTDerror = self.calculateTDError(currentExp)

        self.updateTargetModel()

    def updateTargetModel(self):
        if self.time % self.parameters.TARGET_NETWORK_MAX_STEPS == 0:
            self.network.targetNetwork.set_weights(self.network.valueNetwork.get_weights())

    def decideMove(self, newState, bot):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.network.epsilon:
            newActionIdx = numpy.random.randint(len(self.network.getActions()))
            if __debug__:
                print("Explore")
                bot.setExploring(True)
        else:
            if bot.parameters.NEURON_TYPE == "MLP" or not bot.parameters.EXP_REPLAY_ENABLED:
                # Take action based on greediness towards Q values
                q_Values = self.network.predict(newState)
            else:
                if self.time % self.parameters.UPDATE_LSTM_MOVE_NETWORK:
                    update = True
                else:
                    update = False
                q_Values = self.network.predict_single_trace_LSTM(newState, update)
            newActionIdx = numpy.argmax(q_Values)
            # Book keeping:
            self.qValues.append(q_Values[newActionIdx])
            if __debug__:
                bot.setExploring(False)

        if __debug__ and self.parameters.NEURON_TYPE == "MLP":
            q_values_of_state = self.network.predict(newState)
            average_value = round(numpy.mean(q_values_of_state), 1)
            q_value = round(q_values_of_state[newActionIdx], 1)
            print("Expected Q-value: ", average_value, " Q(s,a) of current action: ", q_value)
            print("")
        newAction = self.network.actions[newActionIdx]
        return newActionIdx, newAction

    def save(self, path):
        self.network.saveModel(path)

    def getNetwork(self):
        return self.network

    def getQValues(self):
        return self.qValues

    def getTDError(self):
        return self.latestTDerror