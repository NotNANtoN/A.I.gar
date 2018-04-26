import numpy
import heapq
import math

class ExpectedSarsa(object):
    def __repr__(self):
        return "Expected Sarsa"

    def __init__(self, numOfNNbots, numOfHumans, network, parameters):
        self.network = network
        self.num_NNbots = numOfNNbots
        self.num_humans = numOfHumans
        self.parameters = parameters

        self.discount = self.parameters.DISCOUNT
        self.rewards = []
        self.actionIdxHistory = []
        self.stateHistory = []
        self.time = 0
        self.terminalTime = math.inf
        self.n = self.parameters.TD + 1
        self.tau = -self.n
        self.latestTDError = None
        self.qValues = []

    def reset(self):
        self.latestTDError = None
        self.time = 0
        self.terminalTime = math.inf
        self.tau = -self.n
        self.stateHistory = []
        self.actionIdxHistory = []
        self.rewards = []
        self.qValues = []

    def load(self, modelName):
        if modelName is not None:
            self.network.load(modelName)

    def testNetwork(self, bot, newState):
        self.network.setEpsilon(0)

        player = bot.getPlayer()
        return self.decideMove(newState, player)

    def learn(self, bot, newState):
        player = bot.getPlayer()
        alive = player.getIsAlive()
        # self.actionIdxHistory.append(bot.getCurrentActionIdx())
        if self.time == 0:
            self.stateHistory.append(newState)

        newActionIdx = None
        newAction = None
        newLastMemory = None
        while(self.tau < self.terminalTime -1):
            if self.time < self.terminalTime:

                # Observe R_n, S_n+1 (except at t == 0 where there is no R and S = S_initial)
                if self.time > 0:
                    newLastMemory = None
                    # Take action A_n (bot just took it through model update)
                    self.stateHistory.append(newState)
                    self.rewards.append(bot.getCumulativeReward())
                    if len(self.rewards) > self.n:
                        del self.rewards[0]
                        del self.stateHistory[0]
                        del self.actionIdxHistory[0]

                # If state is terminal
                if not alive:
                    self.terminalTime = self.time + 1
                else:
                    # Decide on which new action A_n given S_n, only if the player is still alive
                    newActionIdx, newAction = self.decideMove(newState, player)
                    self.actionIdxHistory.append(newActionIdx)

            self.tau = self.time - self.n + 1

            # Given S_n, A_n, R_n, and S_n+1 decide on A_n+1 and train
            # Make sure it's not the first state of the episode
            if self.tau >= 0:
                # Only load memories if expReplay is enabled
                if False: #bot.getExpRepEnabled():
                    memories = bot.getMemories()
                    lastMemory = bot.getLastMemory()

                    td_error, q_value_action, newLastMemory = self.train(self.stateHistory[1], self.stateHistory[0],
                                                          self.actionIdxHistory[0], alive, player, memories, lastMemory)
                else:

                    td_error, q_value_action, newLastMemory = self.train(self.stateHistory[1], self.stateHistory[0],
                                                          self.actionIdxHistory[0], alive, player)
                self.latestTDError = td_error
                self.qValues.append(q_value_action)
            self.time += 1
            if alive:
                break
        return newLastMemory, newActionIdx, newAction

    def train(self, state_1, state_0, currentActionIdx, alive, player, memories=None, lastMemory=None):
        input, target, td_error, q_value_action = self.createInputOutputPair(state_0, currentActionIdx,
                                                                             state_1, player, True)
        newLastMemory = None
        if memories is not None:
            newLastMemory = self.experienceReplay(state_0, state_1, td_error, currentActionIdx, alive,
                                                  player, memories, lastMemory)
        # Fit value network using only the current experience
        else:
            self.network.valueNetwork.train_on_batch(input, target)

        if __debug__ and player.getSelected():
            updatedQvalueOfAction = self.network.valueNetwork.predict(numpy.array([state_0]))[0][
                currentActionIdx]
            print("Qvalue of action after training: ", round(updatedQvalueOfAction, 4))
            print("(also after experience replay, so last shown action is not necessarily this action )")
            print("TD-Error: ", td_error)
            print("")
        # Update the target network after 1000 steps
        # Save the weights of the model when updating the target network to avoid losing progress on program crashes
        self.network.targetNetworkSteps -= 1
        if self.network.targetNetworkSteps == 0:
            self.network.targetNetwork.set_weights(self.network.valueNetwork.get_weights())
            # Added num_humans to the following line
            self.network.targetNetworkSteps = self.network.targetNetworkMaxSteps * (self.num_NNbots + self.num_humans)

        return td_error, q_value_action, newLastMemory

    def decideMove(self, newState, player):
        if player.getIsAlive():
            # Take random action with probability 1 - epsilon
            if numpy.random.random(1) < self.network.epsilon:
                newActionIdx = numpy.random.randint(len(self.network.getActions()))
                if __debug__:
                    player.setExploring(True)
            else:
                if self.parameters.USE_POLICY_NETWORK:
                    numpyNewState = numpy.array([newState])
                    qValues = self.network.valueNetwork.predict(numpyNewState)
                    qValueSum = sum(qValues)
                    normalizedQValues = numpy.array([qValue / qValueSum for qValue in qValues])
                    self.network.policyNetwork.train_on_batch(numpyNewState, normalizedQValues)
                    actionValues = self.network.policyNetwork.predict(numpyNewState)
                    newActionIdx = numpy.argmax(actionValues)
                else:
                    # Take action based on greediness towards Q values
                    qValues = self.network.valueNetwork.predict(numpy.array([newState]))
                    newActionIdx = numpy.argmax(qValues)
                    if __debug__:
                        player.setExploring(False)
            newAction = self.network.actions[newActionIdx]
            return newActionIdx, newAction
        return None, None

    def createInputOutputPair(self, state_0, actionIdx, state_1, player, verbose=False):
        state_Q_values = self.network.getValueNetwork().predict(numpy.array([state_0]))[0]
        target = self.calculateTarget(state_1)
        q_value_of_action = state_Q_values[actionIdx]
        td_error = target - q_value_of_action
        if __debug__ and player.getSelected() and verbose:
            print("")
            # print("State to be updated: ", oldState)
            print("Action: ", self.network.getActions()[actionIdx])
            print("G: ", round(sum(self.rewards), 2))
            # print("S\': ", newState)
            print("Qvalue of action before training: ", round(state_Q_values[actionIdx], 4))
            print("Target Qvalue of that action: ", round(target, 4))
            print("All qvalues: ", numpy.round(state_Q_values, 3))
            print("Expected Q-value: ", round(max(state_Q_values), 3))
            print("TD-Error: ", td_error)
        if self.parameters.USE_TARGET:
            state_Q_values[actionIdx] = target
        else:
            state_Q_values[actionIdx] = td_error
        return numpy.array([state_0]), numpy.array([state_Q_values]), td_error, q_value_of_action

    def calculateTarget(self, state_1):
        targetNetworkEnabled = True
        target = 0
        for i in range(self.tau + 1, min(self.tau + self.n, self.terminalTime)):
            idx = i - self.tau - 1
            target += math.pow(self.discount, idx) * self.rewards[idx]
        if self.tau + self.n < self.terminalTime:
            # The target is the reward plus the discounted prediction of the value network
            if targetNetworkEnabled:
                action_Q_values = self.network.getTargetNetwork().predict(numpy.array([state_1]))[0]
            else:
                action_Q_values = self.network.getValueNetwork().predict(numpy.array([state_1]))[0]
            newActionIdx = numpy.argmax(action_Q_values)
            branchSum = 0
            ndiscount = math.pow(self.discount, self.n)
            actionExplProb = 0.1 * (1/len(self.network.getActions()))
            mainChoiceProb = 0.9
            for i in range(len(self.network.getActions())):
                branchSum += actionExplProb * ndiscount * action_Q_values[i]
            branchSum += mainChoiceProb * ndiscount * action_Q_values[newActionIdx]
            target += branchSum
        return target

    def experienceReplay(self, oldState, newState, td_error, currentActionIdx, alive, player, memories, lastMemory):
        newLastMemory = None
        if alive:
            newLastMemory = self.storeInMemory(currentActionIdx, self.rewards, oldState, newState, td_error, alive,
                                       memories, lastMemory)
        self.train_on_experience(player, memories)
        return newLastMemory

    def storeInMemory(self, actionIdx, reward, oldState, newState, td_error, alive, memories, lastMemory):
        # Store current state, action, reward, state pair in memory
        # Delete oldest memory if memory is at full capacity
        memoryCapacity = self.parameters.MEMORY_CAPACITY
        if len(memories) > memoryCapacity:
            #if numpy.random.random() > 0.0:
            del memories[-1]
            #else:
            #    self.memories.remove(min(self.memories, key = lambda memory: abs(memory[-1])))
        if alive:
            newMemory = [oldState.tolist(), actionIdx, reward, newState.tolist()]
        else:
            newMemory = [oldState.tolist(), actionIdx, reward, None]
        # Square the TD-error and multiply by minus one, because the heap pops the smallest number
        heapq.heappush(memories, ((td_error * td_error) * -1, newMemory, lastMemory))
        return newMemory

    def train_on_experience(self, player, memories):
        # Fit value network on memories
        len_memory = len(memories)
        if len_memory < self.network.getMemoriesPerUpdate():
            return
        inputSize = self.parameters.STATE_REPR_LEN
        outputSize = self.network.getNumOfActions()
        batch_size = self.network.getMemoriesPerUpdate()
        # Initialize vectors
        inputs = numpy.zeros((batch_size, inputSize))
        targets = numpy.zeros((batch_size, outputSize))
        batch_count = 0
        # Get most surprising memories:
        popped_memories = []
        for idx in range(int(batch_size * 0)):
            # Get the item with highest priority (td-error)
            memory = heapq.heappop(memories)[1]
            input, target, td_error = self.memoryToInputOutput(memory)
            # Set input and target
            inputs[idx] = input
            targets[idx] = target
            # Update td-error for memory
            popped_memories.append(((td_error * td_error) * -1, memory))
            batch_count += 1
        # Put the retrieved memories back in memory
        for poppedMemory in popped_memories:
            heapq.heappush(memories, poppedMemory)

        # Get recent memories:
        # WARNING: this does not acutally get the most recent memories, so we have to change it
        # TODO: change it
        for idx in range(int(batch_size * 0)):
            memory = memories[len_memory - idx - 1][1]
            input, target, td_error = self.memoryToInputOutput(memory, player)
            inputs[batch_count] = input
            targets[batch_count] = target
            # Update TD-Error of memory:
            memories[idx] = (td_error, memory)
            batch_count += 1

        # Fill up the rest of the batch with random memories:
        while batch_count < batch_size:
            randIdx = numpy.random.randint(len(memories))
            memory = memories[randIdx][1]
            input, target, td_error = self.memoryToInputOutput(memory, player)
            inputs[batch_count] = input
            targets[batch_count] = target
            # Update TD-Error of memory:
            memories[randIdx] = (td_error, memory)
            batch_count += 1

        # Train on memories
        self.network.trainOnBatch(inputs, targets)

    def memoryToInputOutput(self, memory, player):
        s = memory[0]
        a = memory[1]
        r = memory[2]
        sPrime = memory[3]
        alive = (sPrime is not None)
        return self.createInputOutputPair(s, a, r, sPrime, alive, player)[:-1]

    def getNetwork(self):
        return self.network

    def getQValues(self):
        return self.qValues

    def getTDError(self):
        return self.latestTDError
