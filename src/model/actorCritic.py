import keras
import numpy


class ExpReplay(object):
    # TODO: extend with prioritized replay based on td_error
    def __init(self, parameters):
        self.memories = []
        self.max = parameters.max_memories
        self.batch_size = parameters.memory_batch_size

    def remember(self, old_s, a, r, new_s):
        self.memories.append((old_s, a, r, new_s))


    def canReplay(self):
        return len(self.memories) >= self.batch_size

    def sample(self):
        return numpy.random.sample((self.batch_size, self.memories))


class ValueNetwork(object):
    def __init__(self, parameters):
        if parameters.enable_LSTM:
            #not implemented yet
            quit()
        else:
            self.model = keras.Sequential()

    def predict(self, state):
        return self.model.predict(numpy.array([state]))[0]
        

    def train(self, inputs, targets):
        pass


class PolicyNetwork(object):
    def __init__(self, parameters):
        if parameters.enable_LSTM:
            # not implemented yet
            quit()
        else:
            self.model = keras.Sequential()

    def predict(self, state):
        pass

    def train(self, inputs, targets):
        pass


class Actor(object):
    def __init__(self, parameters):
        self.policyNetwork = PolicyNetwork(parameters)

    def predict(self, state):
        return self.policyNetwork.predict(state)

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
        self.actor = Actor(parameters)
        self.critic = Critic(parameters)
        self.parameters = parameters
        if parameters.enable_exp_rep:
            self.expReplay = ExpReplay(parameters)

    def remember(self, old_s, a, r, new_s):
        pass

    def train(self):
        if self.parameters.enable_exp_rep:
            memories = self.expReplay.sample()
            inputs_critic = []
            targets_critic = []
            total_weight_changes_actor = 0
            for memory in memories:
                # Calculate input and target for critic
                old_s, a, r, new_s = memory
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



    def act_test(self, state):
        return self.actor.predict(state)

    def act_train(self, state):
        actions = self.actor.predict(state)
        std_dev = self.parameters.std_dev
        actions = [numpy.random.normal(mean, std_dev) for mean in actions]
        numpy.clip(actions, 0, 1)
        return actions




