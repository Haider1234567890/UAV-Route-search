import numpy as np
import pandas as pd
import ast


class Agent:
    # learning_rate: learning rate, reward_decay: discount factor, epsilon: epsilon-greedy factor, weather: 'normal' or 'sunny'
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.01, weather='normal', cell_weather=None, env=None):
        self.lr = learning_rate
        self.gamma = reward_decay
        self.actions = actions
        self.epsilon = epsilon
        self.weather = weather
        self.cell_weather = cell_weather if cell_weather is not None else {}
        self.env = env  # Environment reference for collision detection
        # q_table stores Q-values for state-action pairs
        self.q_table = pd.DataFrame(columns=self.actions)

    def update_q_table(self, s, a, r, sig):
        # Normalize states to canonical grid cell keys so the table always stays 144+1 rows
        state = self._normalize_state(s)
        next_state = self._normalize_state(sig)
        self.check_in_qtable(state)
        if next_state != 'finished':
            self.check_in_qtable(next_state)

        reward = r
        cell_type = 'cloudy'
        col, row = None, None

        # Extract grid position from normalized state
        if next_state != 'finished':
            try:
                coords = ast.literal_eval(next_state)
                center_x = (coords[0] + coords[2]) / 2.0
                center_y = (coords[1] + coords[3]) / 2.0
                gridWidth = self.env.gridWidth if self.env else 80
                col = int(center_x // gridWidth)
                row = int(center_y // gridWidth)
                cell_type = self.cell_weather.get((col, row), 'cloudy')
            except Exception:
                pass

        # Composite reward adjustment combining buildings, weather, and birds
        # Base reward r already includes: step cost, goal reward, hazard penalty, building penalty
        
        # 1. Weather-based adjustments (applied to all rewards)
        if cell_type == 'sunny' and reward > 0:
            # Sunny weather boosts positive rewards (e.g., reaching goal)
            reward = int(reward * 1.5)
        elif cell_type == 'snow':
            # Snow applies penalty (worst weather)
            reward = reward - 5
        elif cell_type == 'rain':
            # Rain applies smaller penalty (better than snow, worse than cloudy)
            reward = reward - 2
        # cloudy: no change (neutral weather)
        
        # 2. Additional proximity-based risk penalty for birds (optional enhancement)
        if self.env and col is not None and row is not None:
            # Check if agent is near any bird (within Manhattan distance 1)
            bird_positions = self._get_bird_grid_positions()
            for bird_col, bird_row in bird_positions:
                manhattan_dist = abs(col - bird_col) + abs(row - bird_row)
                if manhattan_dist == 1:
                    # Small penalty for being adjacent to bird (soft avoidance)
                    reward = reward - 3
                    break
        
        # 3. Building collision is already in base reward r (from Layout.step)
        #    No additional modification needed here
        
        # Get current q-table value and update
        q_value = self.q_table.loc[state, a]
        if next_state != 'finished':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        self.q_table.loc[state, a] += self.lr * (q_target - q_value)
    
    def _get_bird_grid_positions(self):
        """Get current grid positions of all birds from environment"""
        if not self.env or not hasattr(self.env, 'blacks'):
            return []
        # Convert bird canvas coords to grid positions
        bird_grid_positions = []
        for black_coords in self.env.blackCoors:
            if len(black_coords) >= 2:
                # Take first two coords as representative point
                cx = (black_coords[0] + black_coords[2]) / 2.0 if len(black_coords) >= 4 else black_coords[0]
                cy = (black_coords[1] + black_coords[3]) / 2.0 if len(black_coords) >= 4 else black_coords[1]
                col = int(cx // self.env.gridWidth)
                row = int(cy // self.env.gridWidth)
                bird_grid_positions.append((col, row))
        return bird_grid_positions

    def action_select(self, observation):
        observation = self._normalize_state(observation)
        self.check_in_qtable(observation)
        # Epsilon-greedy strategy: choose best action with probability (1 - epsilon)
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # np.max(state_action) is the maximum value in the state-action row; choose among actions with that value
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # Choose a random action
            action = np.random.choice(self.actions)

        print("Select action: ", ['up', 'down', 'left', 'right'][action])
        return action

    def check_in_qtable(self, state):
        state = self._normalize_state(state)
        # Add state to q_table if not present
        if state not in self.q_table.index:
            # pandas.DataFrame.append is deprecated; use loc assignment to add a new row
            # Assigning to a non-existing index will create a new row with values matching columns
            self.q_table.loc[state] = [0] * len(self.actions)

    def _normalize_state(self, state):
        """Convert any state representation to a canonical grid cell key."""
        if state == 'finished':
            return 'finished'

        coords = None
        if isinstance(state, list) and len(state) == 4:
            coords = state
        elif isinstance(state, str) and state.startswith('['):
            try:
                parsed = ast.literal_eval(state)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
                    coords = list(parsed)
            except Exception:
                coords = None

        if coords is None:
            return str(state)

        gridWidth = self.env.gridWidth if self.env else 80
        center_x = (coords[0] + coords[2]) / 2.0
        center_y = (coords[1] + coords[3]) / 2.0
        col = int(center_x // gridWidth)
        row = int(center_y // gridWidth)
        x0, y0 = col * gridWidth, row * gridWidth
        x1, y1 = (col + 1) * gridWidth, (row + 1) * gridWidth
        # Use fixed one-decimal formatting to match canonical keys like "[0.0, 0.0, 80.0, 80.0]"
        return f"[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]"
