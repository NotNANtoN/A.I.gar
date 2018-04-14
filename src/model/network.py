import heapq
import keras
import numpy
import tensorflow as tf
import importlib.util
from keras.layers import Dense, LSTM, Softmax
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model


from .parameters import *
from .spatialHashTable import spatialHashTable

class Network(object):

    def __init__(self, trainMode, numOfNNbots, numOfHumans, modelName):
        self.trainMode = trainMode

        self.actions = [[x, y, split, eject] for x in [0, 0.5, 1] for y in [0, 0.5, 1] for split in [0, 1] for
                   eject in [0, 1]]
        # Filter out actions that do a split and eject at the same time
        # Also filter eject actions for now
        for action in self.actions[:]:
            if action[2] and action[3] or action[3]:
                self.actions.remove(action)

        self.num_actions = len(self.actions)

        self.parameters = importlib.import_module('.networkParameters', package="model")
        self.loadedModelName = None

        self.num_NNbots = numOfNNbots
        self.num_humans = numOfHumans

        self.stateReprLen = self.parameters.STATE_REPR_LEN

        self.gpus = self.parameters.GPUS

        # Experience replay:
        self.memoryCapacity = self.parameters.MEMORY_CAPACITY
        self.memoriesPerUpdate = self.parameters.MEMORIES_PER_UPDATE  # Must be divisible by 4 atm due to experience replay

        # Q-learning
        self.targetNetworkSteps = self.parameters.TARGET_NETWORK_STEPS
        self.targetNetworkMaxSteps = self.parameters.TARGET_NETWORK_MAX_STEPS
        self.discount = self.parameters.DISCOUNT
        self.epsilon = self.parameters.EPSILON
        self.frameSkipRate = self.parameters.FRAME_SKIP_RATE
        self.gridSquaresPerFov = self.parameters.GRID_SQUARES_PER_FOV  # is modified by the user later on anyways

        # ANN
        self.learningRate = self.parameters.ALPHA
        self.optimizer = self.parameters.OPTIMIZER
        self.activationFuncHidden = self.parameters.ACTIVATION_FUNC_HIDDEN
        self.activationFuncOutput = self.parameters.ACTIVATION_FUNC_OUTPUT

        self.hiddenLayer1 = self.parameters.HIDDEN_LAYER_1
        self.hiddenLayer2 = self.parameters.HIDDEN_LAYER_2
        self.hiddenLayer3 = self.parameters.HIDDEN_LAYER_3

        weight_initializer_range = math.sqrt(6 / (self.stateReprLen + self.num_actions))
        initializer = keras.initializers.RandomUniform(minval=-weight_initializer_range,
                                                       maxval=weight_initializer_range, seed=None)
        if self.gpus > 1:
            with tf.device("/cpu:0"):
                self.valueNetwork = Sequential()
                self.valueNetwork.add(Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                                           bias_initializer=initializer, kernel_initializer=initializer))
                if self.hiddenLayer2 > 0:
                    self.valueNetwork.add(Dense(self.hiddenLayer2, activation=self.activationFuncHidden, bias_initializer=initializer
                              , kernel_initializer=initializer))
                if self.hiddenLayer3 > 0:
                    self.valueNetwork.add(Dense(self.hiddenLayer3, activation=self.activationFuncHidden, bias_initializer=initializer
                                               , kernel_initializer=initializer))
                self.valueNetwork.add(Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                                       , kernel_initializer=initializer))
                self.valueNetwork = multi_gpu_model(self.valueNetwork, gpus=self.gpus)
        else:
            self.valueNetwork = Sequential()
            hidden1 = None
            if self.parameters.NEURON_TYPE == "MLP":
                hidden1 = Dense(self.hiddenLayer1, input_dim=self.stateReprLen, activation=self.activationFuncHidden,
                                          bias_initializer=initializer, kernel_initializer=initializer)
            elif self.parameters.NEURON_TYPE == "LSTM":
                hidden1 = LSTM(self.hiddenLayer1, input_shape=(self.stateReprLen, 1), activation=self.activationFuncHidden,
                                bias_initializer=initializer, kernel_initializer=initializer)

            self.valueNetwork.add(hidden1)
            #self.valueNetwork.add(Dropout(0.5))
            hidden2 = None
            if self.hiddenLayer2 > 0:
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden2 = Dense(self.hiddenLayer2, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden2 = LSTM(self.hiddenLayer2, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.valueNetwork.add(hidden2)
                #self.valueNetwork.add(Dropout(0.5))

            if self.hiddenLayer3 > 0:
                hidden3 = None
                if self.parameters.NEURON_TYPE == "MLP":
                    hidden3 = Dense(self.hiddenLayer3, activation=self.activationFuncHidden,
                                    bias_initializer=initializer, kernel_initializer=initializer)
                elif self.parameters.NEURON_TYPE == "LSTM":
                    hidden3 = LSTM(self.hiddenLayer3, activation=self.activationFuncHidden,
                                   bias_initializer=initializer, kernel_initializer=initializer)
                self.valueNetwork.add(hidden3)
                #self.valueNetwork.add(Dropout(0.5))

            self.valueNetwork.add(
                Dense(self.num_actions, activation=self.activationFuncOutput, bias_initializer=initializer
                      , kernel_initializer=initializer))

            if self.parameters.USE_POLICY_NETWORK:
                self.policyNetwork = Sequential()
                hidden1 = Dense(50, input_dim=self.stateReprLen, activation='sigmoid',
                                bias_initializer=initializer, kernel_initializer=initializer)
                self.policyNetwork.add(hidden1)
                out = Dense(self.num_actions, activation='softmax', bias_initializer=initializer,
                            kernel_initializer=initializer)
                self.policyNetwork.add(out)


        self.targetNetwork = keras.models.clone_model(self.valueNetwork)
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())



        if self.optimizer == "Adam":
            optimizer = keras.optimizers.Adam(lr=self.learningRate)
        elif self.optimizer == "SGD":
            optimizer = keras.optimizers.SGD(lr=self.learningRate)

        self.valueNetwork.compile(loss='mse', optimizer=optimizer)
        self.targetNetwork.compile(loss='mse', optimizer=optimizer)
        if self.parameters.USE_POLICY_NETWORK:
            self.policyNetwork.compile(loss='mse', optimizer=optimizer)

        if modelName is not None:
            path = "savedModels/" + modelName
            packageName = "savedModels." + modelName
            self.parameters = importlib.import_module('.networkParameters', package=packageName)
            self.loadedModelName = modelName
            self.valueNetwork = load_model(path + "/NN_model.h5")
            self.targetNetwork = self.valueNetwork

    def calculateTarget(self, newState, reward, alive):
        targetNetworkEnabled = True
        target = reward
        if alive:
            # The target is the reward plus the discounted prediction of the value network
            if targetNetworkEnabled:
                action_Q_values = self.targetNetwork.predict(numpy.array([newState]))[0]
            else:
                action_Q_values = self.valueNetwork.predict(numpy.array([newState]))[0]
            newActionIdx = numpy.argmax(action_Q_values)
            target += self.discount * action_Q_values[newActionIdx]
        return target

    def createInputOutputPair(self, oldState, actionIdx, reward, newState, alive, player, verbose=False):
        state_Q_values = self.valueNetwork.predict(numpy.array([oldState]))[0]
        target = self.calculateTarget(newState, reward, alive)
        q_value_of_action = state_Q_values[actionIdx]
        td_error = target - q_value_of_action
        if __debug__ and player.getSelected() and verbose:
            print("")
            # print("State to be updated: ", oldState)
            print("Action: ", self.actions[actionIdx])
            print("Reward: ", round(reward, 2))
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
        return numpy.array([oldState]), numpy.array([state_Q_values]), td_error, q_value_of_action

    def qLearn(self, bot):
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
        if bot.currentAction != None:
            # Get reward of skipped frames
            reward = bot.cumulativeReward
            input, target, td_error, q_value_action = self.createInputOutputPair(bot.oldState, bot.currentActionIdx, reward,
                                                                 newState, alive, bot.player, True)
            # Save data for plotting purposes
            bot.latestTDerror = td_error
            bot.qValues.append(q_value_action)
            # Fit value network using experience replay of random past states:
            if bot.expRepEnabled:
                bot.experienceReplay(reward, newState, td_error)
            # Fit value network using only the current experience
            else:
                self.valueNetwork.train_on_batch(input, target)

            if  __debug__ and bot.player.getSelected():
                updatedQvalueOfAction = self.valueNetwork.predict(numpy.array([bot.oldState]))[0][
                    bot.currentActionIdx]
                print("Qvalue of action after training: ", round(updatedQvalueOfAction, 4))
                print("(also after experience replay, so last shown action is not necessarily this action )")
                print("TD-Error: ", td_error)
                print("")


            # Update the target network after 1000 steps
            # Save the weights of the model when updating the target network to avoid losing progress on program crashes
            self.targetNetworkSteps -= 1
            if self.targetNetworkSteps == 0:
                self.targetNetwork.set_weights(self.valueNetwork.get_weights())
                #Added num_humans to the following line
                self.targetNetworkSteps = self.targetNetworkMaxSteps * (self.num_NNbots + self.num_humans)


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

    def testNetwork(self, bot):
        alive = bot.player.getIsAlive()
        self.epsilon = 0
        if alive:
            newState = bot.getStateRepresentation()
            self.takeAction(newState, bot)

    def takeAction(self, newState, bot):
        # Take random action with probability 1 - epsilon
        if numpy.random.random(1) < self.epsilon:
            bot.currentActionIdx = numpy.random.randint(len(bot.actions))
            if __debug__:
                bot.player.setExploring(True)
        else:
            if self.parameters.USE_POLICY_NETWORK:
                numpyNewState = numpy.array([newState])
                qValues = self.valueNetwork.predict(numpyNewState)
                qValueSum = sum(qValues)
                normalizedQValues = numpy.array([qValue / qValueSum for qValue in qValues])
                self.policyNetwork.train_on_batch(numpyNewState, normalizedQValues)
                actionValues = self.policyNetwork.predict(numpyNewState)
                bot.currentActionIdx = numpy.argmax(actionValues)
            else:
                # Take action based on greediness towards Q values
                qValues = self.valueNetwork.predict(numpy.array([newState]))
                bot.currentActionIdx = numpy.argmax(qValues)
                if __debug__:
                    bot.player.setExploring(False)
        bot.currentAction = self.actions[bot.currentActionIdx]
        bot.skipFrames = self.frameSkipRate
        bot.cumulativeReward = 0

    def trainOnBatch(self, inputs, targets):
        self.valueNetwork.train_on_batch(inputs, targets)

    def saveModel(self, path, bot):
        self.targetNetwork.set_weights(self.valueNetwork.get_weights())
        self.targetNetwork.save(path + bot.type + "_model.h5")


    def setEpsilon(self, val):
        self.epsilon = val

    def getTrainMode(self):
        return self.trainMode

    def getMemoriesPerUpdate(self):
        return self.memoriesPerUpdate

    def getParameters(self):
        return self.parameters

    def getNumOfActions(self):
        return self.num_actions

    def getEpsilon(self):
        return self.epsilon

    def getDiscount(self):
        return self.discount

    def getFrameSkipRate(self):
        return self.frameSkipRate

    def getGridSquaresPerFov(self):
        return self.gridSquaresPerFov

    def getTargetNetworkMaxSteps(self):
        return self.targetNetworkMaxSteps

    def getStateReprLen(self):
        return self.stateReprLen

    def getHiddenLayer1(self):
        return self.hiddenLayer1

    def getHiddenLayer2(self):
        return self.hiddenLayer2

    def getHiddenLayer3(self):
        return self.hiddenLayer3

    def getNumActions(self):
        return self.num_actions

    def getLearningRate(self):
        return self.learningRate

    def getActivationFuncHidden(self):
        return self.activationFuncHidden

    def getActivationFuncOutput(self):
        return self.activationFuncOutput

    def getOptimizer(self):
        return self.optimizer

    def getLoadedModelName(self):
        return self.loadedModelName



