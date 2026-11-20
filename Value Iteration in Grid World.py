
# Practical 10: Value Iteration in Grid World

# Grid size
ROWS = 4
COLS = 4

# Start and Goal positions
START = (0, 0)
GOAL = (3, 3)

# Obstacles
OBSTACLES = [(1, 1), (2, 1)]

# Actions
ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# Parameters
GAMMA = 0.9
THETA = 0.001

# Rewards
GOAL_REWARD = 10
STEP_REWARD = -1
OBSTACLE_REWARD = -5

# Initialize values
V = {}

for r in range(ROWS):
    for c in range(COLS):
        V[(r, c)] = 0

# Check valid position
def valid_position(position):

    r, c = position

    return (
        0 <= r < ROWS and
        0 <= c < COLS and
        position not in OBSTACLES
    )

# Get next state
def get_next_state(state, action):

    dr, dc = ACTIONS[action]

    next_state = (state[0] + dr, state[1] + dc)

    if not valid_position(next_state):
        return state

    return next_state

# Value Iteration
print("Starting Value Iteration...")

while True:

    delta = 0

    for state in V:

        if state == GOAL:
            continue

        old_value = V[state]

        best_value = -float("inf")

        for action in ACTIONS:

            next_state = get_next_state(state, action)

            if next_state == GOAL:
                reward = GOAL_REWARD
            else:
                reward = STEP_REWARD

            new_value = reward + GAMMA * V[next_state]

            if new_value > best_value:
                best_value = new_value

        V[state] = best_value

        delta = max(delta, abs(old_value - V[state]))

    if delta < THETA:
        break

# Extract optimal policy
policy = {}

for state in V:

    if state == GOAL:
        policy[state] = "GOAL"
        continue

    best_action = None
    best_value = -float("inf")

    for action in ACTIONS:

        next_state = get_next_state(state, action)

        if next_state == GOAL:
            reward = GOAL_REWARD
        else:
            reward = STEP_REWARD

        value = reward + GAMMA * V[next_state]

        if value > best_value:
            best_value = value
            best_action = action

    policy[state] = best_action

# Display values and policy
print("\nValue Table:")

for r in range(ROWS):
    for c in range(COLS):
        state = (r, c)

        if state in OBSTACLES:
            print("  WALL ", end=" ")
        else:
            print(f"{V[state]:6.2f}", end=" ")

    print()

print("\nOptimal Policy:")

for r in range(ROWS):
    for c in range(COLS):
        state = (r, c)

        if state in OBSTACLES:
            print("  #  ", end=" ")
        elif state == GOAL:
            print("  G  ", end=" ")
        else:
            print(f"{policy[state]:5s}", end=" ")

    print()

# Follow optimal path
print("\nOptimal Path:")

current = START
path = [current]

for step in range(ROWS * COLS):

    if current == GOAL:
        break

    action = policy[current]
    current = get_next_state(current, action)
    path.append(current)

print(path)

if current == GOAL:
    print("\nGoal Reached Successfully!")
else:
    print("\nGoal Not Reached!")
