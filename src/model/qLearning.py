import numpy
import heapq

class QLearn(object):
    def __init__(self, numOfNNbots, numOfHumans, network, parameters):
        self.num_NNbots = numOfNNbots
        self.num_humans = numOfHumans
        self.name = "Q-learning"
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

    def testNetwork(self, bot, newState):
        self.network.setEpsilon(0)
        player = bot.getPlayer()
        return self.decideMove(newState, player)

    def calculateTarget(self, newState, reward, alive):
        target = reward
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            action_Q_values = self.network.predict_target_network(newState)
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.network.getDiscount() * action_Q_values[newActionIdx]
        return target

    def calculateTDError(self, experience):
        old_s, a, r, new_s = experience
        alive = new_s is not None
        state_Q_values = self.network.predict(old_s)
        target = self.calculateTarget(new_s, r, alive)
        q_value_of_action = state_Q_values[a]
        td_error = target - q_value_of_action
        return td_error

    def train(self, batch):
        batch_len = len(batch)
        inputs = numpy.zeros((batch_len, self.input_len))
        targets = numpy.zeros((batch_len, self.output_len))
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s = sample
            # No new state: dead
            alive = (new_s is not None)
            target = self.calculateTarget(new_s, r, alive)
            inputs[sample_idx] = old_s
            targets[sample_idx] = target
        self.network.trainOnBatch(inputs, targets)


    def learn(self, batch):
        self.time += 1 * self.parameters.FRAME_SKIP_RATE

        self.train(batch)

        #Book keeping. batch[-1] is the current experience:
        self.latestTDerror = self.calculateTDError(batch[-1])

        self.updateTargetModel()

    def updateTargetModel(self):
        if self.time % self.parameters.TARGET_NETWORK_MAX_STEPS == 0:
            self.network.targetNetwork.set_weights(self.network.valueNetwork.get_weights())

    def decideMove(self, newState, player):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.network.epsilon:
            newActionIdx = numpy.random.randint(len(self.network.getActions()))
            if __debug__:
                player.setExploring(True)
        else:
            # Take action based on greediness towards Q values
            q_Values = self.network.predict(newState)
            newActionIdx = numpy.argmax(q_Values)
            # Book keeping:
            self.qValues.append(q_Values[newActionIdx])
            if __debug__:
                player.setExploring(False)
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