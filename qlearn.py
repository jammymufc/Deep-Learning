import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# Grid configuration
BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
START = (1, 0)
OBSTACLES = [(2, 2), (2, 3), (2, 4), (3, 2)]
SPECIAL_JUMP_STATE = (1, 3)
SPECIAL_JUMP_DESTINATION = (3, 3)
SPECIAL_JUMP_REWARD = 5
GOAL_REWARD = 10
STEP_REWARD = -1


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        for obs in OBSTACLES:
            self.board[obs] = -1
        self.state = state
        self.isEnd = False
        self.special_jump_occurred = False

    def giveReward(self):
        if self.special_jump_occurred:
            return SPECIAL_JUMP_REWARD
        elif self.state == WIN_STATE:
            return GOAL_REWARD
        else:
            return STEP_REWARD

    def isEndFunc(self):
        if self.state == WIN_STATE:
            self.isEnd = True

    def nxtPosition(self, action):
        """Get next position based on action"""
        # Reset special jump flag
        self.special_jump_occurred = False

        # Check for special jump
        if self.state == SPECIAL_JUMP_STATE:
            self.special_jump_occurred = True
            return SPECIAL_JUMP_DESTINATION

        # Regular movement
        if action == "North":
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == "South":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "West":
            nxtState = (self.state[0], self.state[1] - 1)
        else:  # East
            nxtState = (self.state[0], self.state[1] + 1)

        # Check if valid move
        if (0 <= nxtState[0] < BOARD_ROWS and
                0 <= nxtState[1] < BOARD_COLS and
                nxtState not in OBSTACLES):
            return nxtState

        # Invalid move, stay in current position
        return self.state

    def showBoard(self):
        """Display the board"""
        for i in range(BOARD_ROWS):
            print('---------------------')
            out = '| '
            for j in range(BOARD_COLS):
                if (i, j) == self.state:
                    token = '*'
                elif (i, j) in OBSTACLES:
                    token = 'X'
                elif (i, j) == WIN_STATE:
                    token = '#'
                elif (i, j) == SPECIAL_JUMP_STATE:
                    token = 'J'
                else:
                    token = '-'
                out += token + ' | '
            print(out)
        print('---------------------')


class Agent:
    def __init__(self, learning_rate=0.23, gamma=0.9, epsilon=0.3):
        self.states = []
        self.actions = ["North", "South", "West", "East"]
        self.State = State()
        self.lr = learning_rate
        self.gamma = gamma
        self.exp_rate = epsilon

        # Initialize state values
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if (i, j) not in OBSTACLES:
                    self.state_values[(i, j)] = 0

    def chooseAction(self):
        """Choose action using epsilon-greedy policy"""
        # Exploration
        if np.random.uniform(0, 1) <= self.exp_rate:
            return np.random.choice(self.actions)

        # Exploitation - choose action with highest expected value
        mx_nxt_reward = float('-inf')
        actions = []

        for a in self.actions:
            nxtPos = self.State.nxtPosition(a)
            nxt_reward = self.state_values[nxtPos]

            if nxt_reward > mx_nxt_reward:
                actions = [a]
                mx_nxt_reward = nxt_reward
            elif nxt_reward == mx_nxt_reward:
                actions.append(a)

        return np.random.choice(actions)

    def takeAction(self, action):
        """Take action and return new state"""
        position = self.State.nxtPosition(action)
        new_state = State(state=position)
        new_state.special_jump_occurred = self.State.special_jump_occurred
        return new_state

    def reset(self):
        """Reset state for new episode"""
        self.states = []
        self.State = State()

    def play(self, rounds=50):
        """Train the agent for specified number of rounds"""
        i = 0
        while i < rounds:
            if self.State.isEnd:
                # Get final reward
                reward = self.State.giveReward()
                self.state_values[self.State.state] = reward
                print(f"Game End Reward {reward}")

                # Back-propagate values
                next_s = self.State.state
                for s in reversed(self.states):
                    if s in OBSTACLES:
                        continue
                    v_current = self.state_values[s]
                    v_next = self.state_values[next_s]
                    updated_value = v_current + self.lr * (v_next - v_current)
                    self.state_values[s] = round(updated_value, 3)
                    next_s = s

                self.reset()
                i += 1
            else:
                # Choose and take action
                action = self.chooseAction()
                next_position = self.State.nxtPosition(action)
                self.states.append(next_position)

                # Print debug info if needed
                # print(f"Current: {self.State.state}, Action: {action}")

                # Handle special jump
                special_jump = self.State.state == SPECIAL_JUMP_STATE
                if special_jump:
                    print(f"SPECIAL JUMP: {SPECIAL_JUMP_STATE} -> {SPECIAL_JUMP_DESTINATION} with reward +5")

                # Take action
                self.State = self.takeAction(action)

                # TD update
                reward = self.State.giveReward()
                prev_state = self.states[-1]
                if prev_state not in OBSTACLES:
                    v_current = self.state_values[prev_state]
                    v_next = self.state_values[self.State.state]
                    updated_value = v_current + self.lr * (reward + self.gamma * v_next - v_current)
                    self.state_values[prev_state] = round(updated_value, 3)

                # Check for end state
                self.State.isEndFunc()
                # print(f"Next: {self.State.state}")
                # print("---------------------")

    def showValues(self):
        """Print state values in text format"""
        for i in range(BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                out += str(self.state_values.get((i, j), 0)).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def showBoardValues(self):
        """Print detailed state values with color coding"""
        print("\nLearned State Values:\n")
        for i in range(BOARD_ROWS):
            print('------------------------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                if (i, j) in OBSTACLES:
                    token = '  X   '
                elif (i, j) == START:
                    token = "{:<6.2f}".format(self.state_values.get((i, j), 0))
                elif (i, j) == WIN_STATE:
                    token = "{:<6.2f}".format(self.state_values.get((i, j), 0))
                elif (i, j) == SPECIAL_JUMP_STATE:
                    token = "{:<6.2f}".format(self.state_values.get((i, j), 0))
                else:
                    token = "{:<6.2f}".format(self.state_values.get((i, j), 0))
                out += token + ' | '
            print(out)
        print('------------------------------------------------\n')


def plot_value_grid(values, obstacles, title="Value Grid"):
    """Create visualization of state values"""
    grid = np.zeros((BOARD_ROWS, BOARD_COLS))
    mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
    bg_colors = np.ones((BOARD_ROWS, BOARD_COLS, 3))  # All white

    # Set special cell colors
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if (i, j) == START:
                bg_colors[i, j] = [0, 1, 0]  # Green
            elif (i, j) == WIN_STATE:
                bg_colors[i, j] = [0, 1, 1]  # Cyan
            elif (i, j) == SPECIAL_JUMP_STATE:
                bg_colors[i, j] = [1, 0, 1]  # Magenta

    # Mark obstacles and build value grid
    for i, j in obstacles:
        mask[i, j] = True
        bg_colors[i, j] = [0, 0, 0]  # Black

    for (i, j), val in values.items():
        grid[i, j] = val

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(bg_colors, origin='upper')

    # Create heatmap mask to exclude special cells
    heatmap_mask = np.zeros_like(grid, dtype=bool)
    for i, j in obstacles + [START, WIN_STATE, SPECIAL_JUMP_STATE]:
        heatmap_mask[i, j] = True
    grid_masked = np.ma.masked_array(grid, mask=heatmap_mask)

    # Apply heatmap overlay
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "cyan"])
    ax.imshow(grid_masked, cmap=cmap, interpolation='nearest', origin='upper',
              vmin=np.min(grid), vmax=np.max(grid), alpha=0.6)

    # Add text values
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if mask[i, j]:
                ax.text(j, i, 'X', va='center', ha='center', color='red', fontweight='bold')
            else:
                val = grid[i, j]
                ax.text(j, i, '{:.2f}'.format(val), va='center', ha='center',
                        color='black', fontweight='bold')

    # Grid lines
    ax.set_xticks(np.arange(BOARD_COLS))
    ax.set_yticks(np.arange(BOARD_ROWS))
    ax.set_xticks(np.arange(-0.5, BOARD_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_ROWS, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.tick_params(which='major', length=0)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def compare_learning_rates():
    """Train agents with different learning rates and compare results"""
    learning_rates = [0.1, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for lr in learning_rates:
        print(f"\nTraining with learning rate = {lr}")
        agent = Agent(learning_rate=lr)
        agent.play(rounds=50)  # Train for 50 rounds

        # Store results
        results[lr] = (agent, agent.state_values)

        # Show state values
        print(f"\nState values for learning rate = {lr}:")
        agent.showBoardValues()

        # Plot grid
        plot_value_grid(agent.state_values, OBSTACLES, f"Value Grid (Learning Rate = {lr})")

    # Find best learning rate
    best_lr = None
    best_score = float('-inf')

    for lr, (agent, values) in results.items():
        # Score based on value of START state (higher is better)
        start_value = values.get(START, 0)
        if start_value > best_score:
            best_score = start_value
            best_lr = lr

    print(f"\nBest learning rate based on start state value: {best_lr}")

    # Compare all learning rates in one grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, lr in enumerate(learning_rates):
        if i < len(axes):
            agent, values = results[lr]

            # Create grid visualization
            grid = np.zeros((BOARD_ROWS, BOARD_COLS))
            for (x, y), val in values.items():
                grid[x, y] = val

            # Plot
            im = axes[i].imshow(grid, cmap='viridis')
            axes[i].set_title(f"Learning Rate = {lr}")

            # Add text values
            for x in range(BOARD_ROWS):
                for y in range(BOARD_COLS):
                    if (x, y) in OBSTACLES:
                        axes[i].text(y, x, 'X', ha='center', va='center', color='red')
                    elif (x, y) in values:
                        val = values[(x, y)]
                        axes[i].text(y, x, f"{val:.2f}", ha='center', va='center',
                                     color='white' if val < 5 else 'black')

    # Hide unused subplot if any
    if len(learning_rates) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

    return results[best_lr][0]  # Return best agent


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("Q-LEARNING WITH VARIABLE LEARNING RATES")
    print("=" * 80)

    # Option 1: Train single agent
    agent = Agent(learning_rate=0.23)  # Use learning rate from reference code
    agent.play(rounds=50)
    print(agent.showValues())
    agent.State.showBoard()
    agent.showBoardValues()
    plot_value_grid(agent.state_values, OBSTACLES)

    # Option 2: Compare different learning rates
    # Uncomment below to run the comparison
    best_agent = compare_learning_rates()