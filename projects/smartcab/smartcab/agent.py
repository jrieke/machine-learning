import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


color_to_index = {'red': 0, 'green': 1}
direction_to_index = {'forward': 0, 'left': 1, 'right': 2, None: 3}
index_to_direction = {0: 'forward', 1: 'left', 2: 'right', 3: None}


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon=0.2, alpha=0.3, gamma=0.85):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.Q = np.random.rand(3, 2, 4, 4, 4, 4)  # table with Q values for all combinations of state (first 5 dimensions; note that next_waypoint cannot have None value) and action (last dimension)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.reset()

        # print "LearningAgent.__init__(): Initialized agent with epsilon = {}, alpha = {}, gamma = {}".format(self.epsilon, self.alpha, self.gamma)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None
        self.inputs = None

    def update(self, t):

        if t == 0:
            # Gather inputs for initial state
            self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
            self.inputs = self.env.sense(self)

            # State is the tuple (next_waypoint, light, oncoming, left, right), where strings are replaced by integer indices. 
            self.state = (direction_to_index[self.next_waypoint], color_to_index[self.inputs['light']], direction_to_index[self.inputs['oncoming']], direction_to_index[self.inputs['left']], direction_to_index[self.inputs['right']])   
        else:
            self.state = self.next_state


        deadline = self.env.get_deadline(self)
        
        
        # TODO: Select action according to your policy
        # action = np.random.choice([None, 'forward', 'left', 'right'])

        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(4)
        else:
            action_index = np.argmax(self.Q[self.state])
        action = index_to_direction[action_index]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, self.inputs, action, reward)  # [debug]

        # Gather inputs for next state
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.inputs = self.env.sense(self)

        self.next_state = (direction_to_index[self.next_waypoint], color_to_index[self.inputs['light']], direction_to_index[self.inputs['oncoming']], direction_to_index[self.inputs['left']], direction_to_index[self.inputs['right']])   

        # TODO: Learn policy based on state, action, reward
        self.Q[self.state][action_index] += self.alpha * (reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.state][action_index])
        # print np.mean(self.Q)

        # print t


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, epsilon=0.05, alpha=0.1, gamma=0.6)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, live_plot=True, update_delay=0.2, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials


def grid_search():
    """Perform a grid search over the parameters of Q learning."""

    # Parameters for grid search. 
    epsilons = np.arange(0., 0.45, 0.05)
    alphas = np.arange(0., 0.75, 0.05)
    gammas = np.arange(0.5, 1.05, 0.05)

    results = pd.DataFrame()

    for epsilon in epsilons:
        for alpha in alphas:
            for gamma in gammas:

                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, epsilon=epsilon, alpha=alpha, gamma=gamma)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, live_plot=False, update_delay=0., display=False)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                sim.run(n_trials=100)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

                # Calculate means of the metrics and store them in a data frame, append parameters. 
                means = pd.DataFrame(sim.rep.summary()).mean(axis=1)
                means['epsilon'] = epsilon
                means['alpha'] = alpha
                means['gamma'] = gamma
                results = results.append(means, ignore_index=True)

    # Save all results to CSV.
    results.to_csv('results/mean_results.csv')

    # Print parameters for highest success rate. 
    print 'Best result:'
    print results.loc[results['success'].idxmax()]

    # Make 3D plot of parameter space. 
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(results['epsilon'], results['alpha'], results['gamma'], s=200*results['success']**2, c=results['success'], vmin=0, vmax=1)
    ax.set_xlabel('epsilon')
    ax.set_ylabel('alpha')
    ax.set_zlabel('gamma')
    fig.colorbar(scat, label='Success rate', pad=0.15)

    plt.savefig('results/parameter_space.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    run()
