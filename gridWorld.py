import numpy as np
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

init()

# global variables
BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
#LOSE_STATE = (1, 3)
START = (1, 0)
DETERMINISTIC = True
OBSTACLES = [(2, 2), (2, 3), (2, 4), (3, 2)]
# Added special jump state and destination
SPECIAL_JUMP_STATE = (1, 3)
SPECIAL_JUMP_DESTINATION = (3, 3)
SPECIAL_JUMP_REWARD = 5


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        for obs in OBSTACLES:
            self.board[obs] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

        # Track if a special jump has occurred in this step
        self.special_jump_occurred = False

    def giveReward(self):
        if self.special_jump_occurred:
            return SPECIAL_JUMP_REWARD
        elif self.state == WIN_STATE:
            return 10
        #elif self.state == LOSE_STATE:
            #return -1
        else:
            return -1  # All other actions result in a -1 reward

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: North, South, West, East
        -------------
        0 | 1 | 2| 3| 4| 5|
        1 |
        2 |
        3 |
        4 |
        5 |
        return next position
        """
        # Reset special jump flag
        self.special_jump_occurred = False
        
        if self.determine:
            if action == "North":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "South":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "West":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            
            # Check for special jump
            if self.state == SPECIAL_JUMP_STATE:
                self.special_jump_occurred = True
                return SPECIAL_JUMP_DESTINATION
                
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS - 1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS - 1)):
                    if nxtState not in OBSTACLES:
                        return nxtState
        return self.state


    def showBoard(self):
        for i in range(BOARD_ROWS):
            print('---------------------')
            out = '| '
            for j in range(BOARD_COLS):
                if (i, j) == self.state:
                    token = Fore.GREEN + '*' + Style.RESET_ALL
                elif (i, j) in OBSTACLES:
                    token = Fore.RED + 'X' + Style.RESET_ALL
                elif (i, j) == (4, 4):
                    token = Fore.CYAN + '#' + Style.RESET_ALL
                elif (i, j) == (1, 3):
                    token = Fore.MAGENTA + 'J' + Style.RESET_ALL
                else:
                    token = Fore.YELLOW + '-' + Style.RESET_ALL
                out += token + ' | '
            print(out)
        print('---------------------')


# Agent of player

class Agent:
    
    def showBoardValues(self):
        print("\nLearned State Values:\n")
        for i in range(BOARD_ROWS):
            print('------------------------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                if (i, j) in OBSTACLES:
                    token = Fore.RED + '  X   ' + Style.RESET_ALL
                elif (i, j) == START:
                    token = Fore.GREEN + "{:<6.2f}".format(self.state_values.get((i, j), 0)) + Style.RESET_ALL
                elif (i, j) == WIN_STATE:
                    token = Fore.CYAN + "{:<6.2f}".format(self.state_values.get((i, j), 0)) + Style.RESET_ALL
                elif (i, j) == SPECIAL_JUMP_STATE:
                    token = Fore.MAGENTA + "{:<6.2f}".format(self.state_values.get((i, j), 0)) + Style.RESET_ALL
                else:
                    token = "{:<6.2f}".format(self.state_values.get((i, j), 0))
                out += token + ' | '
            print(out)
        print('------------------------------------------------\n')


    def __init__(self):
        self.states = []
        self.actions = ["North", "South", "West", "East"]
        self.State = State()
        self.lr = 0.23
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if (i, j) not in OBSTACLES:
                    self.state_values[(i, j)] = 0


    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # Create new state with position
        new_state = State(state=position)
        # Pass the special jump flag from the current state
        new_state.special_jump_occurred = self.State.special_jump_occurred
        return new_state

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print("Game End Reward", reward)
                # back propagate reward (example update)
                next_s = self.State.state
                for s in reversed(self.states):
                    if s in OBSTACLES:
                        continue
                    v_current = self.state_values[s]
                    v_next = self.state_values[next_s]
                    updated_value = v_current + self.lr * (v_next - v_current)
                    self.state_values[s] = round(updated_value, 3)
                    next_s = s  # update for next step
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                next_position = self.State.nxtPosition(action)
                self.states.append(next_position)
                print("current position {} action {}".format(self.State.state, action))
                
                # Check if special jump will occur
                special_jump = self.State.state == SPECIAL_JUMP_STATE
                if special_jump:
                    print("SPECIAL JUMP: [1,3] -> [3,3] with reward +5")
                
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)

                # Immediate TD update after action
                reward = self.State.giveReward()
                prev_state = self.states[-1]
                if prev_state not in OBSTACLES:
                    v_current = self.state_values[prev_state]
                    v_next = self.state_values[self.State.state]
                    updated_value = v_current + self.lr * (reward + v_next - v_current)
                    self.state_values[prev_state] = round(updated_value, 3)
                
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values.get((i, j), 0)).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def plotStateValues(self):
        grid = np.zeros((BOARD_ROWS, BOARD_COLS))
        mask = np.zeros_like(grid, dtype=bool)

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if (i, j) in OBSTACLES:
                    grid[i, j] = np.nan  # Show obstacles as blank
                    mask[i, j] = True
                else:
                    grid[i, j] = self.state_values.get((i, j), 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='black')  # For obstacles

        c = ax.imshow(grid, cmap=cmap, interpolation='nearest', origin='lower')

        # Annotate cells
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if not mask[i, j]:
                    val = grid[i, j]
                    ax.text(j, i, "{:.1f}".format(val), ha='center', va='center', color='white')
                else:
                    ax.text(j, i, "X", ha='center', va='center', color='red')

        # Special markers
        #ax.text(START[1], START[0], "S", ha='center', va='center', color='lime', fontweight='bold')
        #ax.text(WIN_STATE[1], WIN_STATE[0], "G", ha='center', va='center', color='cyan', fontweight='bold')
        #ax.text(SPECIAL_JUMP_STATE[1], SPECIAL_JUMP_STATE[0], "J", ha='center', va='center', color='magenta',
                #fontweight='bold')

        ax.set_title("Learned State Values Heatmap")
        ax.set_xticks(np.arange(BOARD_COLS))
        ax.set_yticks(np.arange(BOARD_ROWS))
        ax.invert_yaxis()
        plt.colorbar(c, ax=ax)
        plt.tight_layout()
        plt.show()


def plot_value_grid(values, obstacles, title="Value Grid"):
    grid = np.zeros((BOARD_ROWS, BOARD_COLS))
    mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    # Base background: all white
    bg_colors = np.ones((BOARD_ROWS, BOARD_COLS, 3))

    # Special tiles
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if (i, j) == START:
                bg_colors[i, j] = [0, 1, 0]  # Green
            elif (i, j) == WIN_STATE:
                bg_colors[i, j] = [0, 1, 1]  # Cyan
            elif (i, j) == SPECIAL_JUMP_STATE:
                bg_colors[i, j] = [1, 0, 1]  # Magenta

    # Mark obstacles
    for i, j in obstacles:
        mask[i, j] = True
        bg_colors[i, j] = [0, 0, 0]  # Pure black

    # Fill value grid
    for (i, j), val in values.items():
        grid[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 6))

    # Background layer: white/specials/black
    ax.imshow(bg_colors, origin='lower')

    # Heatmap: white to cyan (transparent over obstacles)
    cmap = mcolors.LinearSegmentedColormap.from_list("white_to_cyan", ["#ffffff", "#00ffff"])
    norm = mcolors.Normalize(vmin=np.min(grid), vmax=np.max(grid))
    masked_grid = np.ma.masked_where(mask, grid)
    heat = ax.imshow(masked_grid, cmap=cmap, norm=norm, origin='upper', alpha=0.6)

    # Add text
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
    ax.set_xticklabels(np.arange(BOARD_COLS))
    ax.set_yticklabels(np.arange(BOARD_ROWS))
    ax.set_xticks(np.arange(-0.5, BOARD_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_ROWS, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.tick_params(which='major', length=0)

    # Colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='State Value')

    ax.set_title(title)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    ag = Agent()
    ag.play(50)
    print(ag.showValues())
    ag.State.showBoard()
    ag.showBoardValues()
    #ag.plotStateValues()
    plot_value_grid(ag.state_values, OBSTACLES)

