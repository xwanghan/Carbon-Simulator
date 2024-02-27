
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")
    )
)
from Carbon_simulator.foundation import landmarks, resources

def vis_world_frames(dense_logs, lens=None, start=0):
    if lens == None:
        lens = len(dense_logs["world"])
    figs = []
    for i in range(lens):
        figs.append(plot_log_world(dense_logs, i+start))
    return figs


def plot_log_world(dense_log, t, ax=None, remap_key=None):
    maps = dense_log["world"][t]
    states = dense_log["states"][t]

    n_agents = len(states) - 1
    locs = []
    for i in range(n_agents):
        r, c = states[str(i)]["loc"]
        locs.append([r, c])

    if remap_key is None:
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        key_val = np.array(
            [dense_log["states"][0][str(i)][remap_key] for i in range(n_agents)]
        )
        cmap_order = np.argsort(key_val).tolist()

    skills = {}
    for i in range(n_agents):
        skills[i] = {"skill1": states[str(i)]['Manufacture_volume'],
                     "skill2": states[str(i)]['Research_ability']}
    skill1 = sorted(skills, key=lambda k: skills[k]['skill1'])
    skill2 = sorted(skills, key=lambda k: skills[k]['skill2'])

    return plot_world(maps, locs, skill1, skill2, ax, cmap_order)


def plot_world(maps, locs, skill1, skill2, ax=None, cmap_order=None):
    world_size = np.array(maps.get("Carbon_project")).shape
    max_health = {"Carbon_project": 1, "Green_project": 1, "Property": 1, "Carbon_pollution": 1}
    n_agents = len(locs)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax.cla()
    tmp = np.zeros((3, world_size[0], world_size[1]))
    cmap = sns.cubehelix_palette(rot=0.5, hue=2, reverse=True, as_cmap=False, n_colors=len(skill1) + 1)[1:]

    if cmap_order is None:
        cmap_order = list(range(n_agents))
    else:
        cmap_order = list(cmap_order)
        assert len(cmap_order) == n_agents

    scenario_entities = [k for k in maps.keys() if "source" not in k.lower()]
    for entity in scenario_entities:
        if entity in ("Property", "Green_project", "Carbon_pollution"):
            continue
        elif resources.has(entity):
            if resources.get(entity).collectible:
                map_ = (
                        resources.get(entity).color[:, None, None]
                        * np.array(maps.get(entity))[None]
                )
                map_ /= max_health[entity]
                tmp += map_
        elif landmarks.has(entity):
            map_ = (
                    landmarks.get(entity).color[:, None, None]
                    * np.array(maps.get(entity))[None]
            )
            tmp += map_
        else:
            continue

    if isinstance(maps, dict):
        house_idx = np.array(maps.get("Property")["owner"])
        house_health = np.array(maps.get("Property")["health"])
    else:
        house_idx = maps.get("Property", owner=True)
        house_health = maps.get("Property")
    for i in range(n_agents):
        houses = house_health * (house_idx == cmap_order[i])
        agent = np.zeros_like(houses)
        agent += houses
        col = np.array(cmap[i])
        map_ = col[:, None, None] * agent[None]
        tmp += map_

    if isinstance(maps, dict):
        project_idx = np.array(maps.get("Green_project")["owner"])
        project_health = np.array(maps.get("Green_project")["health"])
    else:
        project_idx = maps.get("Green_project", owner=True)
        project_health = maps.get("Green_project")
    for i in range(n_agents):
        projects = project_health * (project_idx == cmap_order[i])
        agent = np.zeros_like(projects)
        agent += projects
        col = np.array(cmap[i])
        map_ = col[:, None, None] * agent[None]
        tmp += map_

    tmp *= 0.6
    tmp = 0.9 - tmp

    tmp = np.transpose(tmp, [1, 2, 0])
    tmp = np.minimum(tmp, 1.0)

    im = ax.imshow(tmp, vmax=1.0, aspect="auto")

    bbox = ax.get_window_extent()

    for i in range(n_agents):
        r, c = locs[cmap_order[i]]
        col = 0.9 - np.array(cmap[i]) * 0.6
        ax.plot(c, r, "o", markersize=int(bbox.height * 8 * (1 + skill1[i] * 0.3)) / 550 + 2.5, color="black")
        ax.plot(c, r, "o", markersize=int(bbox.height * 8 * (1 + skill1[i] * 0.3)) / 550, color=col)

    for y in range(project_health.shape[0]):
        for x in range(project_health.shape[1]):
            if project_health[y, x]:
                vertices = np.array([[x, y - 0.5], [x + 0.5, y], [x, y + 0.5], [x - 0.5, y]])
                ax.fill(vertices[:, 0], vertices[:, 1], color="y", alpha=0.4)

    pollution_idx = np.array(maps.get("Carbon_pollution"))
    for y in range(pollution_idx.shape[0]):
        for x in range(pollution_idx.shape[1]):
            if pollution_idx[y, x]:
                vertices = np.array([[x, y - 0.5], [x + 0.5, y + 0.5], [x - 0.5, y + 0.5]])
                ax.fill(vertices[:, 0], vertices[:, 1], color="r", alpha=0.2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig