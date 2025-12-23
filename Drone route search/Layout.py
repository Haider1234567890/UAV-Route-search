import numpy as np
import tkinter as tk
import random

class InitLayout(tk.Tk):
    # gridNum: number of grid cells, gridWidth: width of each cell, objWidth: width of objects inside a cell
    def __init__(self, gridNum=12, gridWidth=80, objWidth=50, num_buildings=22, start_pos=(0,0), goal_pos=(9,9), weather='normal', cell_weather=None):
        super(InitLayout, self).__init__()
        self.weather = weather
        # Generate weather distribution if not provided
        if cell_weather is None:
            cell_weather = {}
            # Left half: sunny
            for col in range(gridNum // 2):
                for row in range(gridNum):
                    cell_weather[(col, row)] = 'sunny'
            # Right half: cloudy (default)
            for col in range(gridNum // 2, gridNum):
                for row in range(gridNum):
                    cell_weather[(col, row)] = 'cloudy'
            # Top-right quadrant: snow
            for col in range(gridNum // 2, gridNum):
                for row in range(0, gridNum // 2):
                    cell_weather[(col, row)] = 'snow'
            # Bottom-left quadrant: rain
            for col in range(0, gridNum // 2):
                for row in range(gridNum // 2, gridNum):
                    cell_weather[(col, row)] = 'rain'
        self.cell_weather = cell_weather
        if self.weather == 'sunny':
            self.title('Drone route planning - Sunny')
        else:
            self.title('Drone route planning')
        self.gridNum = gridNum
        self.gridWidth = gridWidth
        self.objWidth = objWidth
        self.borderSize = self.gridNum * self.gridWidth
        # 4 actions
        self.action_space = ['up', 'down', 'left', 'right']
        self.actions_num = len(self.action_space)
        # Black hole positions, indices in row/column grid (static hazards)
        self.blacks = [[0, 2], [1, 2], [3, 0], [4, 4]]  # static hazards
        # Black hole canvas coordinate list
        self.blackCoors = []
        # Store start and goal positions
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        # Generate buildings randomly, avoiding start, goal, and black hole positions
        all_cells = [(col, row) for col in range(gridNum) for row in range(gridNum)]
        forbidden_cells = set([start_pos, goal_pos] + [tuple(b) for b in self.blacks])
        available_cells = [cell for cell in all_cells if cell not in forbidden_cells]
        self.buildings = random.sample(available_cells, min(num_buildings, len(available_cells)))
        # Canvas coordinate list for buildings
        self.buildingCoors = []
        self.start_drawing()
  
 
    def start_drawing(self):
        # Initialize canvas
        self.drawing = tk.Canvas(self, height=self.borderSize, width=self.borderSize, bg='#c9ccd0')
        # Draw cell backgrounds for local weather
        for col in range(self.gridNum):
            for row in range(self.gridNum):
                cell_type = self.cell_weather.get((col, row), self.weather)
                if cell_type == 'sunny':
                    color = '#FFA500'
                elif cell_type == 'snow':
                    color = '#E6F7FF'  # light icy blue for snowy cells
                elif cell_type == 'rain':
                    color = '#B0C4DE'  # light steel blue for rainy cells
                else:
                    color = '#c9ccd0'
                x0 = col * self.gridWidth
                y0 = row * self.gridWidth
                x1 = x0 + self.gridWidth
                y1 = y0 + self.gridWidth
                self.drawing.create_rectangle(x0, y0, x1, y1, fill=color, outline='')
                # If snowy, draw a few small snowflakes (simple circles)
                if cell_type == 'snow':
                    for _ in range(3):
                        fx = random.uniform(x0 + 5, x1 - 5)
                        fy = random.uniform(y0 + 5, y1 - 5)
                        r = 3
                        self.drawing.create_oval(fx - r, fy - r, fx + r, fy + r, fill='white', outline='')
                # If rainy, draw small raindrops (vertical lines)
                elif cell_type == 'rain':
                    for _ in range(4):
                        rx = random.uniform(x0 + 5, x1 - 5)
                        ry = random.uniform(y0 + 5, y1 - 15)
                        self.drawing.create_line(rx, ry, rx, ry + 8, fill='#4682B4', width=1)
        # Draw grid lines
        for col in range(0, self.borderSize, self.gridWidth):
            x0, y0, x1, y1 = col, 0, col, self.borderSize
            self.drawing.create_line(x0, y0, x1, y1)
        for row in range(0, self.borderSize, self.gridWidth):
            x0, y0, x1, y1 = 0, row, self.borderSize, row
            self.drawing.create_line(x0, y0, x1, y1)

        # Initial origin point
        start_point = [0, 0]
        start_pos = np.array([self.gridWidth / 2, self.gridWidth / 2])
        start_pos = start_pos + np.array([self.gridWidth * start_point[0], self.gridWidth * start_point[1]])

        # Loop to draw black holes
        for index, pos_tup in enumerate(self.blacks):
            col, row = pos_tup
            center_pos = start_pos + np.array([self.gridWidth * col, self.gridWidth * row])
            # Draw a lightning-shaped polygon centered at the cell
            cx, cy = center_pos[0], center_pos[1]
            size = self.objWidth * 0.8
            # Points for a simple bird-shaped polygon (relative to center)
            points = [
                cx - size*0.5, cy + size*0.1,
                cx - size*0.15, cy - size*0.25,
                cx + size*0.2, cy - size*0.1,
                cx + size*0.45, cy - size*0.02,
                cx + size*0.2, cy + size*0.15,
                cx - size*0.15, cy + size*0.05,
            ]
            bird_id = self.drawing.create_polygon(points, fill='#6b4423', outline='black', width=2)
            self.blackCoors.append(self.drawing.coords(bird_id))
            if not hasattr(self, 'black_ids'):
                self.black_ids = []
            self.black_ids.append(bird_id)
        # Draw static buildings (obstacles)
        if not hasattr(self, 'building_ids'):
            self.building_ids = []
        for bidx, bpos in enumerate(self.buildings):
            bcol, brow = bpos
            bcenter = start_pos + np.array([self.gridWidth * bcol, self.gridWidth * brow])
            # draw a filled rectangle to represent the building
            half = self.objWidth / 2
            bid = self.drawing.create_rectangle(
                bcenter[0] - half, bcenter[1] - half,
                bcenter[0] + half, bcenter[1] + half,
                fill='#7f7f7f', outline='black')
            self.building_ids.append(bid)
            self.buildingCoors.append(self.drawing.coords(bid))

        # Draw the destination
        # Destination grid index (col, row)
        gold = list(self.goal_pos)
        dist_center = start_pos + np.array([self.gridWidth * gold[0], self.gridWidth * gold[1]])
        self.dist = self.drawing.create_oval(
            dist_center[0] - self.objWidth / 2, dist_center[1] - self.objWidth / 2,
            dist_center[0] + self.objWidth / 2, dist_center[1] + self.objWidth / 2,
            fill='blue')

        # Draw the agent
        self.rect = self.drawing.create_oval(
            start_pos[0] - self.objWidth / 2, start_pos[1] - self.objWidth / 2,
            start_pos[0] + self.objWidth / 2, start_pos[1] + self.objWidth / 2,
            fill='green')

        self.drawing.pack()
        # Black holes now move randomly every 500ms
        self.move_blacks()
 
    def reset(self):
        # Redraw agent position
        self.drawing.delete(self.rect)
        origin = np.array([self.gridWidth / 2, self.gridWidth / 2])
        self.rect = self.drawing.create_oval(
            origin[0] - self.objWidth / 2, origin[1] - self.objWidth / 2,
            origin[0] + self.objWidth / 2, origin[1] + self.objWidth / 2,
            fill='green')

        return self.drawing.coords(self.rect)

    # 智能体移动
    def step(self, action):
        s = self.drawing.coords(self.rect)
        # Determine agent current grid index from canvas coords
        center_x = (s[0] + s[2]) / 2.0
        center_y = (s[1] + s[3]) / 2.0
        cur_col = int(center_x // self.gridWidth)
        cur_row = int(center_y // self.gridWidth)

        # Determine target grid index based on action
        tgt_col, tgt_row = cur_col, cur_row
        if action == 0:  # up
            tgt_row = max(0, cur_row - 1)
        elif action == 1:  # down
            tgt_row = min(self.gridNum - 1, cur_row + 1)
        elif action == 2:  # left
            tgt_col = max(0, cur_col - 1)
        elif action == 3:  # right
            tgt_col = min(self.gridNum - 1, cur_col + 1)

        # If target cell is a static building, block movement
        if (tgt_col, tgt_row) in self.buildings:
            # No movement into buildings; small penalty
            sig = self.drawing.coords(self.rect)
            reward = -5
            finished = False
            return sig, reward, finished

        # Compute pixel offset and move the agent
        dx = (tgt_col - cur_col) * self.gridWidth
        dy = (tgt_row - cur_row) * self.gridWidth
        self.drawing.move(self.rect, dx, dy)
        sig = self.drawing.coords(self.rect)
 
        # Default reward and done flag
        reward = -1
        finished = False
        # Reached destination
        if sig == self.drawing.coords(self.dist):
            reward = 100
            finished = True
            sig = 'finished'
        # Fell into a black hole
        elif sig in self.blackCoors:
            reward = -100
            finished = True
        # Check collision with black holes
        for black_coords in self.blackCoors:
            if sig == black_coords:
                reward = -100
                finished = True
                return sig, reward, finished
 
        return sig, reward, finished
 
    def render(self):
        self.update()

    def move_blacks(self):
        # Each black hole moves randomly one cell in a direction (up/down/left/right); reverse if out of bounds
        import random
        for idx, black_id in enumerate(self.black_ids):
            dx, dy = 0, 0
            direction = random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up':
                dy = -self.gridWidth
            elif direction == 'down':
                dy = self.gridWidth
            elif direction == 'left':
                dx = -self.gridWidth
            elif direction == 'right':
                dx = self.gridWidth

            # Get current coordinates
            coords = self.drawing.coords(black_id)
            # Check bounds, reverse direction if out of bounds
            new_x0, new_y0, new_x1, new_y1 = coords[0]+dx, coords[1]+dy, coords[2]+dx, coords[3]+dy
            if new_x0 < 0 or new_y0 < 0 or new_x1 > self.borderSize or new_y1 > self.borderSize:
                dx, dy = -dx, -dy

            # Move the lightning (black hole)
            self.drawing.move(black_id, dx, dy)
            # Update blackCoors
            self.blackCoors[idx] = self.drawing.coords(black_id)

        # Move every 500ms
        self.after(500, self.move_blacks)
