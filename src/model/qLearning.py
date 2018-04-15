import numpy
import heapq

class QLearn(object):

    def __init__(self, numOfNNbots, numOfHumans):
        self.network = None
        self.num_NNbots = numOfNNbots
        self.num_humans = numOfHumans

    def setNetwork(self, network):
        self.network = network

    def testNetwork(self, bot):
        alive = bot.player.getIsAlive()
        self.network.setEpsilon(0)
        if alive:
            newState = bot.getStateRepresentation()
            self.takeAction(newState, bot)

    def takeAction(self, newState, bot):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.network.epsilon:
            bot.currentActionIdx = numpy.random.randint(len(bot.actions))
            if __debug__:
                bot.player.setExploring(True)
        else:
            if self.network.getParameters().USE_POLICY_NETWORK:
                numpyNewState = numpy.array([newState])
                qValues = self.network.valueNetwork.predict(numpyNewState)
                qValueSum = sum(qValues)
                normalizedQValues = numpy.array([qValue / qValueSum for qValue in qValues])
                self.network.policyNetwork.train_on_batch(numpyNewState, normalizedQValues)
                actionValues = self.network.policyNetwork.predict(numpyNewState)
                bot.currentActionIdx = numpy.argmax(actionValues)
            else:
                # Take action based on greediness towards Q values
                qValues = self.network.valueNetwork.predict(numpy.array([newState]))
                bot.currentActionIdx = numpy.argmax(qValues)
                if __debug__:
                    bot.player.setExploring(False)
        bot.currentAction = self.network.actions[bot.currentActionIdx]
        bot.skipFrames = self.network.frameSkipRate
        bot.cumulativeReward = 0

    def learn(self, bot):
        #After S has been initialized, set S as oldState and take action A based on policy
        alive = bot.player.getIsAlive()

        bot.cumulativeReward += bot.getReward() if bot.lastMass else 0
        bot.lastReward = bot.cumulativeReward

        if alive:
            bot.rewardAvgOfEpisode = (bot.rewardAvgOfEpisode * bot.rewardLenOfEpisode + bot.lastReward)\
                                      / (bot.rewardLenOfEpisode + 1)
            bot.rewardLenOfEpisode += 1
        # Do not train if we are skipping this frame
        if bot.skipFrames > 0 :
            bot.skipFrames -= 1
            bot.currentAction[2:4] = [0, 0]
            bot.latestTDerror = None
            if alive:
                return

        newState = bot.getStateRepresentation()

        # Only train when we there is an old state to train
        if bot.currentAction is not None:
            # Get reward of skipped frames
            reward = bot.getCumulativeReward()
            oldState = bot.getOldState()
            currentActionIdx = bot.currentActionIdx
            player = bot.getPlayer()
            input, target, td_error, q_value_action = self.network.createInputOutputPair(oldState, currentActionIdx, reward,
                                                                 newState, alive, player, True)
            # Save data for plotting purposes
            bot.latestTDerror = td_error
            bot.qValues.append(q_value_action)
            # Fit value network using experience replay of random past states:
            if bot.expRepEnabled:
                self.experienceReplay(reward, newState, td_error, bot)
            # Fit value network using only the current experience
            else:
                self.network.valueNetwork.train_on_batch(input, target)

            if  __debug__ and bot.player.getSelected():
                updatedQvalueOfAction = self.network.valueNetwork.predict(numpy.array([oldState]))[0][
                    bot.currentActionIdx]
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

        # If the player is alive then save the action, state and mass of this update
        if bot.player.getIsAlive():
            self.takeAction(newState, bot)
            bot.lastMass = bot.player.getTotalMass()
            bot.oldState = newState
        # Otherwise reset values to start a new episode for this actor
        else:
            print(bot.player, " died.")
            print("Average reward of ", bot.player, " for this episode: ", bot.rewardAvgOfEpisode)
            bot.reset()

    def experienceReplay(self, reward, newState, td_error, bot):
        if bot.getPlayer().getIsAlive():
            bot.remember(bot.getOldState(), bot.getCurrentActionIdx(), reward, newState, td_error)
        self.train_on_experience(bot)

    def train_on_experience(self, bot):
        memories = bot.getMemories()
        player = bot.getPlayer()
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
            heapq.heappush(self.memories, poppedMemory)

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
        return self.network.createInputOutputPair(s, a, r, sPrime, alive, player)[:-1]

    def getNetwork(self):
        return self.network
