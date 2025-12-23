import os
import time
import pandas as pd
from Layout import InitLayout
from Agent import Agent

# Absolute directory of this script for consistent file I/O
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

 
def start():

    TOTAL_EXPLORE_EPOCH = 200
    # Total number of episodes/iterations
    # Speed up first 80 epochs (no delay), then slow down for visualization
    FAST_LEARNING_EPOCHS = 80

    # Metrics: total rewards per episode, success count, steps per episode
    episode_rewards = []
    success_count = 0
    episode_steps = []
    success_steps = []

    for epoc in range(TOTAL_EXPLORE_EPOCH):
        # Initial agent observation/position
        observation = env.reset()
        # Total reward and step counter for each episode
        total_reward = 0
        steps = 0

        while True:
            # Update canvas
            env.render()

            action = MyAgent.action_select(str(observation))
            # Move agent and get reward
            next_observation, reward, done = env.step(action)
      
            # Update q-table
            MyAgent.update_q_table(str(observation), action, reward, str(next_observation))
            observation = next_observation

            # Periodic save during long episodes so values stay visible
            if steps % 20 == 0:
                MyAgent.q_table.to_csv(os.path.join(SCRIPT_DIR, "Qtable.csv"))

            # Speed up first 80 epochs, then add delay for visualization
            if epoc >= FAST_LEARNING_EPOCHS:
                time.sleep(0.1)
            # Accumulate total reward and steps
            total_reward = total_reward + reward
            steps += 1

            if done:
                # Record metrics
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                if next_observation == 'finished':
                    success_count += 1
                    success_steps.append(steps)
                print('==== epoch %d R: %.6f, steps: %d ====' %(epoc, total_reward, steps))
                # Save at end of each episode to persist updates
                MyAgent.q_table.to_csv(os.path.join(SCRIPT_DIR, "Qtable.csv"))
                break
    # Summary metrics after training
    avg_reward = sum(episode_rewards)/len(episode_rewards) if episode_rewards else 0.0
    # Standard deviation of rewards (population std)
    reward_std = 0.0
    if episode_rewards:
        mean_r = avg_reward
        reward_std = (sum((r - mean_r)**2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
    avg_steps = sum(episode_steps)/len(episode_steps) if episode_steps else 0.0
    success_rate = success_count / TOTAL_EXPLORE_EPOCH if TOTAL_EXPLORE_EPOCH > 0 else 0.0
    avg_success_steps = (sum(success_steps)/len(success_steps)) if success_steps else 0.0
    print('==== Summary ====')
    print('Episodes:', TOTAL_EXPLORE_EPOCH)
    print('Success rate: %.2f%%' % (success_rate*100))
    print('Average reward: %.3f' % avg_reward)
    print('Average steps: %.2f' % avg_steps)
    print('Reward std: %.3f' % reward_std)
    print('Average steps (success only): %.2f' % avg_success_steps)

    MyAgent.q_table.to_csv(os.path.join(SCRIPT_DIR, "Qtable.csv"))
    env.destroy()

if __name__ == "__main__":
    env = InitLayout()
    MyAgent = Agent(actions=range(env.actions_num), weather=env.weather, cell_weather=env.cell_weather, env=env)
    
    # Ensure Qtable always has the full 12x12 grid states (144) plus 'finished'
    def _load_full_qtable(agent, env):
        states = []
        for c in range(env.gridNum):
            for r in range(env.gridNum):
                x0, y0 = c * env.gridWidth, r * env.gridWidth
                x1, y1 = (c + 1) * env.gridWidth, (r + 1) * env.gridWidth
                states.append(f"[{x0}.0, {y0}.0, {x1}.0, {y1}.0]")
        states.append('finished')

        full_q = pd.DataFrame(0.0, index=states, columns=range(env.actions_num))
        path = os.path.join(SCRIPT_DIR, "Qtable.csv")
        if os.path.exists(path):
            try:
                loaded = pd.read_csv(path, index_col=0)
                loaded = loaded.reindex(columns=range(env.actions_num)).fillna(0.0)
                for idx in loaded.index:
                    if idx in full_q.index:
                        full_q.loc[idx] = loaded.loc[idx]
            except Exception:
                pass

        agent.q_table = full_q

    _load_full_qtable(MyAgent, env)
    env.after(10, start)

    # Start main loop
    env.mainloop()
