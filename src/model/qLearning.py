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

    def learn(self, batch):
        self.steps += 1 * self.parameters.FRAME_SKIP_RATE
        batch_len = len(batch)
        inputs = numpy.zeros((batch_len, self.input_len))
        targets = numpy.zeros((batch_len, self.output_len))
        td_error = 0
        for sample_idx, sample in enumerate(batch):
            old_s, a, r, new_s = sample
            old_s = old_s[0]
            # No new state: dead
            alive = (new_s is not None)
            state_Q_values = self.network.predict(old_s)
            target = self.calculateTarget(new_s, r, alive)
            q_value_of_action = state_Q_values[a]
            td_error = target - q_value_of_action
            inputs[sample_idx] = old_s
            targets[sample_idx] = target

        self.latestTDerror = td_error

        self.network.trainOnBatch(inputs, targets)

        self.updateTargetModel()

    def updateTargetModel(self):
        if self.time % self.parameters.TARGET_NETWORK_MAX_STEPS == 0:
            self.network.targetNetwork.set_weights(self.network.valueNetwork.get_weights())

    def learn_messy(self, bot, newState):
        # Observe R_n, S_n+1
        newLastMemory = None
        reward = bot.getCumulativeReward()
        currentActionIdx = bot.getCurrentActionIdx()
        player = bot.getPlayer()
        alive = player.getIsAlive()
        # Given S_n, A_n, R_n, and S_n+1 decide on A_n+1 and train
        # Make sure it's not the first state of the episode
        if bot.getCurrentAction() is not None:
            # Only load memories if expReplay is enabled
            oldState = bot.getLastState()
            if bot.getExpRepEnabled():
                memories = bot.getMemories()
                lastMemory = bot.getLastMemory()
                td_error, q_value_action, newLastMemory = self.train(newState, reward, oldState,
                                                      currentActionIdx, alive, player, memories, lastMemory)
            else:
                td_error, q_value_action, newLastMemory = self.train(newState, reward, oldState,
                                                      currentActionIdx, alive, player)
            self.latestTDerror = td_error
            self.qValues.append(q_value_action)
        # Decide on which new action A_n given S_n, only if the player is still alive
        newActionIdx, newAction = self.decideMove(newState, player, alive)

        return newLastMemory, newActionIdx, newAction

    def train(self, newState, reward, oldState, currentActionIdx, alive, player, memories=None, lastMemory=None):
        input, target, td_error, q_value_action = self.createInputOutputPair(oldState, currentActionIdx, reward,
                                                                             newState, alive, player, True)
        newLastMemory = None
        if memories is not None:
            newLastMemory = self.experienceReplay(reward, oldState, newState, td_error, currentActionIdx, alive,
                                                  player, memories, lastMemory)
        # Fit value network using only the current experience
        else:
            self.network.valueNetwork.train_on_batch(input, target)

        if  __debug__ and player.getSelected():
            updatedQvalueOfAction = self.network.valueNetwork.predict(numpy.array([oldState]))[0][
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
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.network.epsilon:
            newActionIdx = numpy.random.randint(len(self.network.getActions()))
            if __debug__:
                player.setExploring(True)
        else:
            # Take action based on greediness towards Q values
            q_Values = self.network.valueNetwork.predict(newState)[0]
            newActionIdx = numpy.argmax(q_Values)
            # Book keeping:
            self.qValues.append(q_Values[newActionIdx])
            if __debug__:
                player.setExploring(False)
        newAction = self.network.actions[newActionIdx]
        return newActionIdx, newAction

    def save(self, path):
        self.network.saveModel(path)


    def createInputOutputPair(self, oldState, actionIdx, reward, newState, alive, player, verbose=False):
        state_Q_values = self.network.getValueNetwork().predict(numpy.array([oldState]))[0]
        target = self.calculateTarget(newState, reward, alive)
        q_value_of_action = state_Q_values[actionIdx]
        td_error = target - q_value_of_action
        if __debug__ and player.getSelected() and verbose:
            print("")
            # print("State to be updated: ", oldState)
            print("Action: ", self.network.getActions()[actionIdx])
            print("Reward: ", round(reward, 2))
            # print("S\': ", newState)
            print("Qvalue of action before training: ", round(state_Q_values[actionIdx], 4))
            print("Target Qvalue of that action: ", round(target, 4))
            print("All qvalues: ", numpy.round(state_Q_values, 3))
            print("Expected Q-value: ", round(max(state_Q_values), 3))
            print("TD-Error: ", td_error)
        if self.network.getParameters().USE_TARGET:
            state_Q_values[actionIdx] = target
        else:
            state_Q_values[actionIdx] = td_error
        return numpy.array([oldState]), numpy.array([state_Q_values]), td_error, q_value_of_action

    def experienceReplay(self, reward, oldState, newState, td_error, currentActionIdx, alive, player, memories, lastMemory):
        newLastMemory = None
        if alive:
            newLastMemory = self.storeInMemory(currentActionIdx, reward, oldState, newState, td_error, alive,
                                       memories, lastMemory)
        self.train_on_experience(player, memories)
        return newLastMemory

    def storeInMemory(self, actionIdx, reward, oldState, newState, td_error, alive, memories, lastMemory):
        # Store current state, action, reward, state pair in memory
        # Delete oldest memory if memory is at full capacity
        memoryCapacity = self.network.getMemoryCapacity()
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
        inputSize = self.network.getParameters().STATE_REPR_LEN
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
        return self.latestTDerror
