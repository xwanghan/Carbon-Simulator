from Carbon_simulator.foundation import landmarks, resources
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n_agents = 5

def extract_action(log, step):
    action_data = log["actions"][step]
    actions = {}
    for key in action_data.keys():
        if key != "p":
            if action_data[key] == {}:
                actions[key] = "None"
            elif action_data[key] == {'Carbon_component': 1}:
                actions[key] = "Produce"
            elif action_data[key] == {'Carbon_component': 2}:
                actions[key] = "Investment"
            elif action_data[key] == {'Gather': 1}:
                actions[key] = "Move Left"
            elif action_data[key] == {'Gather': 2}:
                actions[key] = "Move Right"
            elif action_data[key] == {'Gather': 3}:
                actions[key] = "Move Up"
            elif action_data[key] == {'Gather': 4}:
                actions[key] = "Move Down"
            else:
                actions[key] = ''.join([f'{key}{value}' for key, value in action_data[key].items()])
    action_df = pd.DataFrame(list(actions.items()), columns=['Agent', 'Action'])
    return action_df

def plot_skill(log, cmap):
    skills = {}
    states = log["states"][0]
    for i in range(n_agents):
        skills[i] = {"Manufacture_volume": states[str(i)]['Manufacture_volume'],
                     "Research_ability": states[str(i)]['Research_ability']}
    skill1 = sorted(skills, key=lambda k: skills[k]['Manufacture_volume'])
    skill2 = sorted(skills, key=lambda k: skills[k]['Research_ability'])
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bbox = ax.get_window_extent()
    for i in range(n_agents):
        r = skill1[i] + 1
        c = skill2[i] + 1
        col = 0.9 - np.array(cmap[i]) * 0.6
        ax.plot(c, r, "o", markersize=int(bbox.height * 8 * (1 + skill1[i] * 0.3)) / 550 + 2.5, color="black")
        ax.plot(c, r, "o", markersize=int(bbox.height * 8 * (1 + skill1[i] * 0.3)) / 550, color=col)
        ax.text(c, r+0.3, "Agent {}".format(i), fontsize=6, ha='center', va='center', color="black")
    ax.set_facecolor([0.9,0.9,0.9])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 6])
    ax.set_xticklabels(["low", "high"])
    ax.set_yticks([0, 6])
    ax.set_yticklabels(["low", "high"])
    ax.set_xlabel('Manufacture_volume')
    ax.set_ylabel('Research_ability')
    return fig

def accumulate_reward(reward, agent_id):
    reward_list = []
    accumulated_reward = 0
    for x in reward:
        accumulated_reward += x[agent_id]
        reward_list += [accumulated_reward]
    return reward_list
def plot_reward(log, step, cmap):
    reward_log = log["rewards"][:step]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i in range(n_agents):
        ax.plot(
            accumulate_reward(reward_log, str(i)),
            label="agent {}".format(i),
            color=0.9-0.6*np.array(cmap[i]),
            )
    ax.plot(
        accumulate_reward(reward_log, "p"),
        label="planner",
        color="black",
        )
    ax.set_facecolor([0.9, 0.9, 0.9])
    ax.set_title("Rewards")
    ax.set_xlim(-2, 502)
    ax.set_ylim(-50, 1000)
    return fig

def plot_quota(log, step, cmap):
    state_log = log["states"][:step]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i in range(n_agents):
        ax.plot([x[str(i)]["inventory"]['Carbon_idx'] + x[str(i)]["escrow"]['Carbon_idx']
                for x in state_log
                ],
                label=i,
                color=0.9-0.6*np.array(cmap[i]),
                )
    ax.set_facecolor([0.9, 0.9, 0.9])
    ax.set_title("Carbon emission credits")
    ax.set_xlim(-2, 502)
    ax.set_ylim(-5, 20)
    return fig

def plot_coins(log, step, cmap):
    state_log = log["states"][:step]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i in range(n_agents):
        ax.plot([x[str(i)]["inventory"]['Coin'] + x[str(i)]["escrow"]['Coin']
                for x in state_log
                ],
                label=i,
                color=0.9-0.6*np.array(cmap[i]),
                )
    ax.set_facecolor([0.9, 0.9, 0.9])
    ax.set_title("Coins")
    ax.set_xlim(-2, 502)
    ax.set_ylim(-50, 1800)
    return fig

def plot_level(log, step, cmap):
    state_log = log["states"][:step]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i in range(n_agents):
        ax.plot([x[str(i)]['Carbon_emission_rate']
                 for x in state_log
                ],
                label=i,
                color=0.9-0.6*np.array(cmap[i]),
                )
    ax.set_facecolor([0.9, 0.9, 0.9])
    ax.set_title("Carbon emission level")
    ax.set_xlim(-2, 502)
    ax.set_ylim(-0.05, 1.05)
    return fig

def plot_labor(log, step, cmap):
    state_log = log["states"][:step]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i in range(n_agents):
        ax.plot(
            [x[str(i)]["endogenous"]["Labor"] for x in state_log],
            label=i,
            color=0.9-0.6*np.array(cmap[i]),
        )
    ax.set_facecolor([0.9, 0.9, 0.9])
    ax.set_title("Labor")
    ax.set_xlim(-2, 502)
    ax.set_ylim(-0.05, 2000)
    return fig

def plot_trade(log, step, cmap):
    trade_log = log["Trade"][:step]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    x = [0]*10
    over = False
    for i in range(10):
        num = 0
        total_p = 0
        for j in range(i*50, i*50+50):
            if j == step:
                over = True
            if over:
                break
            for dict1 in trade_log[j]:
                if dict1["price"] > 0:
                    num += 1
                    total_p += dict1["price"]
        x[i] = total_p/num if num else 0
        if over:
            break

    ax.bar(
        range(10),
        x,
        color=0.9-0.6*np.array(cmap[0]),
        width=0.5
    )
    ax.set_facecolor([0.9, 0.9, 0.9])
    ax.set_title("Carbon price")
    ax.set_xlim(-0.2, 10.2)
    ax.set_xticks([0, 10])
    ax.set_xticklabels([0, 500])
    ax.set_ylim(0, 20.2)
    return fig