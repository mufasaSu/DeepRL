import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FlowField:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_flow_at_position(self, x, y):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_flow_grid(self, resolution=20):
        ''' Returns a grid of shape (width, height, 2, 2) where the penultimate dimension is the x,y coordinate and the last dimension represents the flow vector at each position.
        The resolution parameter determines how many points are sampled in each direction.'''

        grid_x = np.linspace(0, self.width, resolution)
        grid_y = np.linspace(0, self.height, resolution)
        flow_grid = np.zeros((resolution, resolution, 2, 2))

        for i, x in enumerate(grid_x):
            for j, y in enumerate(grid_y):
                flow_grid[i, j, :, 0] = [x, y]
                flow_grid[i, j, :, 1] = self.get_flow_at_position(x, y)

        return flow_grid


class UniformFlowField(FlowField):
    def __init__(self, height, width, flow_vector):
        super().__init__(height, width)
        self.flow_vector = flow_vector
    
    def get_flow_at_position(self, x, y):
        if 0 <= x <= self.height and 0 <= y <= self.width:
            return self.flow_vector
        else:
            raise ValueError(f"Position ({x}, {y}) is outside the flow field.")

class SegmentedFlowField(UniformFlowField):
    def __init__(self, height, width, flow_vector):
        super().__init__(height, width, flow_vector)

    def get_flow_at_position(self, x, y):
        if 0 <= x <= self.height and 0 <= y <= self.width:
            third_height = self.height // 3
            if third_height <= x < 2 * third_height:
                return self.flow_vector
            else:
                return (0, 0)
        else:
            raise ValueError(f"Position ({x}, {y}) is outside the flow field.")

class SingleGyreFlowField(FlowField):
    def __init__(self, height, width, center, radius, strength):
        super().__init__(height, width)
        self.center = center
        self.radius = radius
        self.strength = strength

    def get_flow_at_position(self, y, x):  # Swap x and y in the parameter list
        if 0 <= y <= self.height and 0 <= x <= self.width:
            dx = x - self.center[1]  # Swap center coordinates
            dy = y - self.center[0]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.radius:
                return (-self.strength * dy, self.strength * dx)
            else:
                return (0, 0)
        else:
            raise ValueError(f"Position ({x}, {y}) is outside the flow field.")  # Swap x and y in the error message





class Environment:
    def __init__(self, flow_field, initial_xy, target, threshold, magnitude,
                 action_type="continous", num_actions=None,
                 save_render_name = None, penalty=True): # initial_xy - list
        self.flow_field = flow_field
        self.target = target
        self.threshold = threshold
        self.initial_state = initial_xy.copy()
        self.current_state = initial_xy.copy()
        self.history = [self.initial_state.copy()]
        self.action_type = action_type
        self.num_actions = num_actions
        self.magnitude = magnitude
        # self.save_render_name = save_render_name
        self.penalty = penalty
        self.reward_history = []

        if self.action_type == "discrete":
            # Define the number of actions
            assert num_actions is not None, "Please specify the number of actions."

            # Calculate the angle of each action
            angles = np.linspace(0, 2 * np.pi, num_actions, endpoint=False)

            # Set the length of each vector
            length = magnitude

            # Calculate the x, y components of each vector
            x = length * np.cos(angles)
            y = length * np.sin(angles)

            self.action_map = {i: (x[i], y[i]) for i in range(num_actions)}
    
    def get_initial_state(self):
        return self.initial_state

    def step(self, action):
        state = self.current_state
        #print("Current state:", state)
        flow = self.flow_field.get_flow_at_position(*self.current_state) # based on current position
        if self.action_type == "continous":
            self.current_state[0] += action[0] + flow[0] # update x
            self.current_state[1] += action[1] + flow[1] # update y
        elif self.action_type == "discrete":
            self.current_state[0] += self.action_map[action][0] + flow[0]
            self.current_state[1] += self.action_map[action][1] + flow[1]
        #print("New state:", self.current_state)
        self.history.append(self.current_state.copy())
        #print("History:", self.history)
        new_state = self.current_state
        reward = self.compute_reward(new_state)
        self.reward_history.append(reward)
        done = self.is_done()

        #self.replay_buffer.add(state, action, new_state, reward, done)
        return new_state, reward, done

    def reset(self):
        self.current_state = self.initial_state.copy()  # Reset agent to initial state
        self.history = [self.initial_state.copy()]
        self.reward_history = []
        # reward = self.compute_reward(self.initial_state[0])
        return self.current_state.copy()

    def compute_reward(self, position):
        # Calculate the Euclidean distance from the current position to the target
        distance = ((position[0] - self.target[0]) ** 2 + (position[1] - self.target[1]) ** 2) ** 0.5
        distance = - distance
        # if agent steps out of field --> add massive penalty
        if self.penalty:
            if not (0 <= self.current_state[0] < self.flow_field.width and 0 <= self.current_state[1] < self.flow_field.height):
                distance -= 100
        # Negative of the distance as the reward
        return distance

    def is_done(self):
        # Check if the agent is outside the grid
        if not (0 <= self.current_state[0] < self.flow_field.width and 0 <= self.current_state[1] < self.flow_field.height):
            outside_of_grid = True
            print("Episode terminated: Agent moved outside the grid.")
            return outside_of_grid
        
        # Calculate the distance between the agent and the target
        distance = ((self.current_state[0] - self.target[0]) ** 2 + (self.current_state[1] - self.target[1]) ** 2) ** 0.5
        reached_goal = (distance <= self.threshold)
        return reached_goal

    def render(self, plot_reward=False): # plot_reward for debugging
        fig, ax = plt.subplots()

        X, Y = np.mgrid[0:self.flow_field.width, 0:self.flow_field.height]
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(self.flow_field.height):  
            for j in range(self.flow_field.width):
                U[i, j], V[i, j] = self.flow_field.get_flow_at_position(j, i)  

        ax.quiver(X, Y, U, V, pivot='mid')

        # # Plot the agent's trajectory
        # x_vals, y_vals = zip(*self.history)
        # ax.plot(x_vals, y_vals, 'ro-')

        # Mark the agent's current position
        ax.plot(self.current_state[0], self.current_state[1], 'bo')

        # Make a big Green X at the target
        ax.plot(self.target[0], self.target[1], 'gx', markersize=8, markeredgewidth=2)

        # Draw a circle to show the threshold
        circle = plt.Circle(self.target, self.threshold, color='g', fill=False)
        ax.add_artist(circle)

        # Add every second reward value on the line between x_vals and y_vals and make it small
        if plot_reward:
            for i in range(0, len(self.reward_history), 2):
                ax.text(x_vals[i], y_vals[i], round(self.reward_history[i], 2), fontsize=8)
       

        # Set the aspect of the plot to be equal and set limits and labels
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, self.flow_field.width)
        ax.set_ylim(0, self.flow_field.height)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        #ax.set_title('Agent Movement in Flow Field')


        # Return the figure and axis for further use
        return fig, ax

