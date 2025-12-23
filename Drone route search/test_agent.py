from Agent import Agent

if __name__ == '__main__':
    actions = [0,1,2,3]
    agent = Agent(actions)
    print('初始 q_table:')
    print(agent.q_table)
    agent.check_in_qtable('state1')
    print('\n添加 state1 后 q_table:')
    print(agent.q_table)
    agent.check_in_qtable('state2')
    print('\n添加 state2 后 q_table:')
    print(agent.q_table)
    # Calling check_in_qtable again on an existing state should not change q_table
    agent.check_in_qtable('state1')
    print('\n再次检查 state1（应与上次相同）:')
    print(agent.q_table)
