import gym, random, tempfile
import numpy as np
from gym import wrappers
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DDQL:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9993
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epochs = 1
        self.verbose = 0
        self.minibatch_size = 30
        self.memory = deque(maxlen=5000)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()

        # Add 2 hidden layers with 64 nodes each
        model.add(Dense(64, input_dim=self.nS, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, s, a, r, s_prime, done):
        self.memory.append((s, a, r, s_prime, done))

    def target_model_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def selectAction(self, s):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA)
        q = self.model.predict(s)
        return np.argmax(q[0])

    def replay(self):
        # Vectorized method for experience replay
        minibatch = random.sample(self.memory, self.minibatch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
            
            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma, \
                        predict_sprime_target[not_done_indices, \
                        np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(self.minibatch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=self.epochs, verbose=self.verbose)

    def replayIterative(self):
        # Iterative method - this performs the same function as replay() but is not vectorized 
        s_list = []
        y_state_list = []
        minibatch = random.sample(self.memory, self.minibatch_size)
        for s, a, r, s_prime, done in minibatch:
            s_list.append(s)
            y_action = r
            if not done:
                y_action = r + self.gamma * np.amax(self.model.predict(s_prime)[0])

            print y_action
            
            y_state = self.model.predict(s)
            y_state[0][a] = y_action
            y_state_list.append(y_state)
        self.model.fit(np.squeeze(s_list), np.squeeze(y_state_list), batch_size=self.minibatch_size, epochs=1, verbose=0)

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    tdir = tempfile.mkdtemp()
    env = gym.make('LunarLander-v2')
    env = wrappers.Monitor(env, tdir, force=True, video_callable=False)

    nS = env.observation_space.shape[0]
    nA = env.action_space.n

    agent = DDQL(nS, nA)

    # Set to true to use saved model
    viewOnly = True

    if viewOnly:
        agent.model.load_weights('./weights/trained_agent.h5')
        episodes = 100
        agent.epsilon = 0
    else:
        episodes = 10000

    # Cumulative reward
    reward_avg = deque(maxlen=100)

    for e in range(episodes):
        episode_reward = 0
        s = env.reset()
        s = np.reshape(s, [1, nS])

        for time in range(1000):
            if viewOnly:
                env.render()

            # Query next action from learner and perform action
            a = agent.selectAction(s)
            s_prime, r, done, info = env.step(a)

            # Add cumulative reward
            episode_reward += r

            # Reshape new state
            s_prime = np.reshape(s_prime, [1, nS])

            # Add experience to memory
            if not viewOnly:
                agent.add_memory(s, a, r, s_prime, done)

            # Set current state to new state
            s = s_prime

            #Perform experience replay if memory length is greater than minibatch length
            if not viewOnly:
                if len(agent.memory) > agent.minibatch_size:
                    agent.replay()

            # If episode is done, exit loop
            if done:
                if not viewOnly:
                    agent.target_model_update()
                break

        # epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Running average of past 100 episodes
        reward_avg.append(episode_reward)
        print 'episode: ', e, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % np.average(reward_avg), ' frames: ', time, ' epsilon: ', '%.2f' % agent.epsilon
        
        # with open('trained_agent.txt', 'a') as f:
        #     f.write(str(np.average(reward_avg)) + '\n')

    env.close()
    gym.upload(tdir, api_key='sk_EJo79Jo0RsSEA4EF6Cp5mg')


#agent.model.save_weights('weights.h5')
#agent.model.load_weights('weights.h5')