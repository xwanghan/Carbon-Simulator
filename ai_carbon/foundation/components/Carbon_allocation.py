# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_carbon.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class CarbonRedistribution(BaseComponent):
    """Redistributes the total coin of the mobile agents as evenly as possible.

    Note:
        If this component is used, it should always be the last component in the order!
    """

    name = "CarbonRedistribution"
    required_entities = ["Carbon_idx", "Carbon_project"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    """
    Required methods for implementing components
    --------------------------------------------
    """

    def __init__(
            self,
            *base_component_args,
            planner_mode="inactive",
            fixed_punishment=100,
            total_idx=200,
            max_year_percent=100,

            predefined = None,

            **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.planner_mode = planner_mode
        assert self.planner_mode in ["inactive", "active"]

        self.total_idx = float(total_idx)
        assert self.total_idx >= 1

        self.step_count = 0

        self.fixed_punishment = int(fixed_punishment)
        assert self.total_idx >= 0

        self.max_year_percent = int(max_year_percent)
        assert 0 <= self.max_year_percent <= 100

        self.predefined = str(predefined)

        self.log = []

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        if agent_cls_name == "BasicPlanner":
            if self.planner_mode == "active":
                ### Action space ###

                return [
                    ("Carbon_{:03d}".format(int(r)), 101)
                    for r in range(self.n_agents + 2)
                ]
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        if agent_cls_name == "BasicMobileAgent":
            return {}
        if agent_cls_name == "BasicPlanner":
            return {"punishment": 0,
                    "year_num": 0,
                    "env_idx": [0 * self._episode_length / self.period],
                    "mobile_idx": [[0 * self._episode_length / self.period] * self.n_agents],
                    "average_Er": 1
                    }
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Redistributes inventory coins so that all agents have equal coin endowment.
        """
        world = self.world
        world.planner.state["year_num"] = self.world.timestep // self.period

        # punishment at end of years#
        if self.world.timestep % self.period == 0:
            for agent in world.agents:
                if agent.state["inventory"]["Carbon_idx"] < 0:
                    punishment = world.planner.state["punishment"] * abs(agent.state["inventory"]["Carbon_idx"])
                    agent.state["inventory"]["Coin"] -= punishment

            sum_Er = 0
            for agent in world.agents:
                sum_Er += agent.state["Carbon_emission_rate"]

            world.planner.state["average_Er"] = sum_Er / world.n_agents
            assert 0 <= world.planner.state["average_Er"] <= 1

            self.log.append({
                "settlement_idx": self.world.planner.state["settlement_idx"],
            })


        # divided idx at start of years#
        elif world.timestep % self.period == 1:
            for agent in world.agents:
                if agent.state["inventory"]["Carbon_idx"] < 0:
                    self.world.planner.state["settlement_idx"][agent.idx] -= agent.state["inventory"]["Carbon_idx"]

            if self.planner_mode == "active":
                idx_action = []
                total_percent = 0
                for i in range(self.n_agents + 2):
                    planner_action = self.world.planner.get_component_action(
                        self.name, "Carbon_{:03d}".format(int(i))
                    )
                    if i == self.n_agents:
                        # world.planner.state["env_idx"] = planner_action
                        total_percent = planner_action
                    elif i == self.n_agents + 1:
                        world.planner.state[
                            "punishment"] = self.fixed_punishment if self.fixed_punishment else planner_action * 10
                    else:
                        # world.planner.state["mobile_idx"][i] = planner_action
                        idx_action.append(planner_action)

                '''if sum(idx_action) > world.planner.state["remained_idx"]:
                    idx_action = [ia * int(world.planner.state["remained_idx"]) // sum(idx_action) for ia in idx_action]'''

                # world.planner.state["env_idx"] = idx_action[-1]
                # env_idx = 10% of this year total idx, this year total idx = self.total_idx * total_percent/100
                year_idx = self.total_idx  * total_percent / 100

                world.planner.state["env_idx"] = int(year_idx / 10)
                for i in range(self.n_agents):
                    # mobile_idx = idx_action[i] // sum(idx_action) * 0.9 * this year total idx
                    if sum(idx_action):
                        world.planner.state["mobile_idx"][i] = int(year_idx * 9 / 10 * idx_action[i] / sum(idx_action))
                    else:
                        world.planner.state["mobile_idx"][i] = int(year_idx * 9 / 10 / self.n_agents)

                world.planner.state["remained_idx"] -= self.world.planner.state["env_idx"] + sum(
                    self.world.planner.state["mobile_idx"])

                for agent in world.agents:
                    agent.state["inventory"]["Carbon_idx"] = world.planner.state["mobile_idx"][agent.idx]
                    agent.state["escrow"]["Carbon_idx"] = 0

            elif self.planner_mode == "inactive":
                if self.predefined == "flat":
                    idx_action = [1, 1, 1, 1, 1]
                    total_percent = 10
                elif self.predefined == "decreasing":
                    total_percents = [16, 16, 14, 12, 10, 10, 8, 6, 4, 4]
                    assert sum(total_percents) == 100, sum(total_percents)
                    idx_action = [5, 5, 4, 3, 3]
                    total_percent = total_percents[world.timestep // self.period]
                elif self.predefined == "convex":
                    total_percents = [6, 8, 10, 12, 14, 14, 12, 10, 8, 6]
                    assert sum(total_percents) == 100, sum(total_percents)
                    idx_action = [5, 5, 4, 3, 3]
                    total_percent = total_percents[world.timestep // self.period]
                else:
                    assert "predefined not in (flat, decreasing, convex)"
                # Divide the Carbon-idx to agents


                year_idx = self.total_idx  * total_percent / 100
                world.planner.state["env_idx"] = int(year_idx / 10)
                for i in range(self.n_agents):
                    # mobile_idx = idx_action[i] // sum(idx_action) * 0.9 * this year total idx
                    if sum(idx_action):
                        world.planner.state["mobile_idx"][i] = int(year_idx * 9 / 10 * idx_action[i] / sum(idx_action))
                    else:
                        world.planner.state["mobile_idx"][i] = int(year_idx * 9 / 10 / self.n_agents)

                sorted_V = sorted(world.agents, key=lambda agent_V: agent_V.state["Manufacture_volume"], reverse=True)
                for agent_idx in range(self.n_agents):
                    sorted_V[agent_idx].state["inventory"]["Carbon_idx"] = world.planner.state["mobile_idx"][agent_idx]
                    sorted_V[agent_idx].state["escrow"]["Carbon_idx"] = 0

            else:
                assert self.planner_mode in ["inactive", "active"]

            # Decided Punishment
            assert world.planner.state["punishment"] >= 0

            # Divide Carbon_idx to env
            assert world.planner.state["env_idx"] >= 0

            # layout the Carbon_project
            '''world.maps.set("Carbon_project", np.zeros(world.world_size))
            world.maps.set("Carbon_projectSourceBlock", np.zeros(world.world_size))'''
            empty = world.maps.empty
            project_num = 0
            n_tries = 0
            while project_num < world.planner.state["env_idx"]:
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
                r = np.random.randint(world.world_size[0])
                c = np.random.randint(world.world_size[1])
                if empty[r, c]:
                    world.maps.set_point("Carbon_project", r, c, 1)
                    world.maps.set_point("Carbon_projectSourceBlock", r, c, 1)
                    empty = world.maps.empty
                    project_num += 1

            self.log.append({
                "env_idx": world.planner.state["env_idx"],
                "punishment": world.planner.state["punishment"],
                "mobile_idx": world.planner.state["mobile_idx"],
                "settlement_idx": self.world.planner.state["settlement_idx"],
            })
        else:
            self.log.append([])

    def generate_observations(self):
        """This component does not add any observations."""
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "year_num": self.world.planner.state["year_num"],
                "average_Er": self.world.planner.state["average_Er"],
            }
        obs_dict[self.world.planner.idx] = {
            "punishment": self.world.planner.state["punishment"],
            "year_num": self.world.planner.state["year_num"],
            "env_idx": self.world.planner.state["env_idx"],
            "mobile_idx": self.world.planner.state["mobile_idx"],
            "agents_Research_ability": [agent.state["Research_ability"] for agent in self.world.agents],
            "agents_volume": [agent.state["Manufacture_volume"] for agent in self.world.agents],
            "agents_emission_rate": [agent.state["Carbon_emission_rate"] for agent in self.world.agents],
            "settlement_idx": self.world.planner.state["settlement_idx"],
            "remained_idx": self.world.planner.state["remained_idx"],
            "average_Er": self.world.planner.state["average_Er"],
        }
        return obs_dict

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        if self.planner_mode == "inactive":
            masks = {}
        elif self.planner_mode == "active":
            masks = super().generate_masks(completions=completions)
            for k, v in masks[self.world.planner.idx].items():
                if self.world.timestep % self.period == 0:
                    remained_idx_mask = np.ones_like(v)
                    remained_idx_mask[0] = 0
                    if k == "Carbon_{:03d}".format(int(self.n_agents)):
                        remained_idx_mask[min(int(self.world.planner.state["remained_idx"]/self.total_idx*100), self.max_year_percent)+1:] = 0
                else:
                    remained_idx_mask = np.zeros_like(v)
                    remained_idx_mask[0] = 1
                '''remained_idx_mask[int(self.world.planner.state["remained_idx"]):] = 0'''
                masks[self.world.planner.idx][k] = remained_idx_mask
        else:
            assert self.planner_mode in ["inactive", "active"]
        return masks

    def additional_reset_steps(self):

        world = self.world
        world.planner.state["punishment"] = self.fixed_punishment if self.fixed_punishment else 100  # 10, 30
        world.planner.state["year_num"] = 0

        world.planner.state["remained_idx"] = float(self.total_idx)

        world.planner.state["settlement_idx"] = np.zeros(self.n_agents)

        self.log = []

        world.planner.state["average_Er"] = 1

        world.planner.state["env_idx"] = 0

        world.planner.state["mobile_idx"] = [0] * self.n_agents

    def get_dense_log(self):
        if self.planner_mode == "inactive":
            return None
        elif self.planner_mode == "active":
            return self.log
