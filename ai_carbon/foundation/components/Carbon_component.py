# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import random

from ai_carbon.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class Carbon_component(BaseComponent):
    """
    Allows mobile agents to build house landmarks in the world using stone and wood,
    earning income.

    Can be configured to include heterogeneous building skill where agents earn
    different levels of income when building.

    Args:
        payment (int): Default amount of coin agents earn from building.
            Must be >= 0. Default is 10.
        payment_max_skill_multiplier (int): Maximum skill multiplier that an agent
            can sample. Must be >= 1.0.0. Default is 1.0.0.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a multiplier of 1.0.0. "pareto" and
            "lognormal" sample skills from the associated distributions.
        require_Carbon_idx (float): Labor cost associated with building a Property.
            Must be >= 0. Default is 10.
        project_gather (float):
        labor (float): Labor cost associated with building a Property.
    """

    name = "Carbon_component"
    component_type = "Carbon_component"
    required_entities = ["Carbon_idx", "Carbon_emission", "Coin", "Property", "Carbon_pollution", "Labor",
                         "Carbon_project", "Green_project"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
            self,
            *base_component_args,
            payment=10,
            require_Carbon_idx=1.0,
            labor=10.0,
            debuff=0.3,
            env_recover_ability=3,
            research_setting=["e^-", 0.5], # ["-log"or"e^-", int]
            labor_multiple=False,
            ability_independent=True,

            evaluate=False,

            lowest_rate=0.1,

            random_fails=0.3,
            delay=3,
            forget=10,

            **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment = int(payment)
        assert self.payment >= 0

        self.require_Carbon_idx = float(require_Carbon_idx)
        assert self.require_Carbon_idx >= 0

        self.labor = float(labor)
        assert self.labor >= 0

        self.debuff = float(debuff)
        assert self.debuff >= 0

        self.env_recover_ability = float(env_recover_ability)
        assert self.env_recover_ability >= 0

        self.research_type = research_setting[0]
        assert self.research_type in ["-log", "e^-"]
        # index of needed Research action to achieve 100% with Ra is 1.0.0
        self.a = research_setting[1]
        assert self.a >= 0

        self.labor_multiple = bool(labor_multiple)

        self.ability_independent = bool(ability_independent)

        self.evaluate = bool(evaluate)

        self.lowest_rate = float(lowest_rate)
        assert 0 <= self.lowest_rate <= 1

        self.random_fails = float(random_fails)
        assert 0 <= self.random_fails <= 1

        self.delay = int(delay)
        assert self.delay >= 0

        self.forget = int(forget)
        assert self.forget >= 0

        self.Manufactures = {"builds": [], "research": []}

    def agent_can_build(self, agent):
        """Return True if agent can actually build in its current location."""
        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return True

    def agent_can_research(self, agent):
        """Return True if agent can actually build in its current location."""
        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return agent.state["inventory"]["Coin"]>0

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        # This component adds 1.0.0 action that mobile agents can take: build a house
        if agent_cls_name == "BasicMobileAgent":
            return 2

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"Manufacture_volume": 1, "Research_ability": 1, "Carbon_emission_rate": 1, "Start_Er": 1,
                    "Research_count": [0, 0], "Research_history": [0] * max(self.delay, self.forget)}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood to house+coin for agents that choose to build and can.
        """
        world = self.world
        builds = []
        research = []
        if self.world.timestep % self.period != 1:

            # Apply any building or research actions taken by the mobile agents
            for agent in world.get_random_order_agents():

                action = agent.get_component_action(self.name)

                # Update the Carbon_emission_rate
                previous_rate = agent.state["Carbon_emission_rate"]
                research_activity = ''
                # forget if more than self.forget time steps do not do research
                if agent.state["Research_count"][1] > 0:
                    if sum(agent.state["Research_history"][:self.forget]) == 0:
                        research_activity += 'Forget'
                        agent.state["Research_count"][0] -= 1
                        agent.state["Research_count"][1] -= 1
                        agent.state["Research_history"][0] = 2
                # Delay research
                if agent.state["Research_history"][self.delay] == 1:
                    research_activity += 'Delayed improve'
                    # Research_count +1.0.0
                    agent.state["Research_count"][0] += 1
                    agent.state["Research_count"][1] += 1

                # Update Emission rate
                if self.research_type == "-log":
                    if agent.state["Start_Er"]==1:
                        agent.state["Carbon_emission_rate"] = max(max(1-(self.world.timestep // self.period + 1)/(self.episode_length / self.period), self.lowest_rate), agent.state["Start_Er"] - np.log(
                            1 + (agent.state["Research_ability"] * agent.state["Research_count"][0])
                        ) / np.log(self.a + 1))
                    else:
                        agent.state["Carbon_emission_rate"] = max(
                            max(1 - (self.world.timestep // self.period + 1) / (self.episode_length / self.period),
                                self.lowest_rate), agent.state["Start_Er"] - np.log(
                                1 + (agent.state["Research_ability"] * agent.state["Research_count"][1])
                            ) / np.log(self.a + 1))

                elif self.research_type == "e^-":
                    assert agent.state["Start_Er"]==1, "Can't set research_type be e^- and tech_share_year be True at same time"
                    power_efficiency = np.e**(-agent.state["Research_ability"] * agent.state["Research_count"][0] * self.a)
                    sum_power_efficiency = np.sum([np.e**(-i_agent.state["Research_count"][0] * self.a) * i_agent.state["Manufacture_volume"] for i_agent in world.agents])
                    green_rate = max(1-np.sum(world.maps.get("Green_project"))/(sum_power_efficiency + np.sum(world.maps.get("Green_project"))), 0)
                    assert 0<= green_rate <=1
                    agent.state["Carbon_emission_rate"] = max(power_efficiency * green_rate, self.lowest_rate)

                # Update Research_history
                agent.state["Research_history"][1:] = agent.state["Research_history"][:-1]
                agent.state["Research_history"][0] = 0

                # This component doesn't apply to this agent!
                if action is None:
                    continue

                # NO-OP!
                if action == 0:
                    pass

                # Build! (If you can.)
                elif action == 1:
                    if self.agent_can_build(agent):

                        # Obtain agent's location
                        loc_r, loc_c = agent.loc

                        # Calculate the Carbon_emission
                        Carbon_emission = self.require_Carbon_idx * agent.state["Manufacture_volume"] * agent.state[
                            "Carbon_emission_rate"]

                        # Get debuff
                        for public, health in world.location_public(loc_r, loc_c).items():
                            if public == "Carbon_pollution" and health >= 1:
                                Carbon_emission = (1 + self.debuff) * Carbon_emission

                        # Update Carbon_idx
                        agent.state["inventory"]["Carbon_idx"] -= Carbon_emission

                        # Place a Property where the agent is standing
                        world.create_landmark("Property", loc_r, loc_c, agent.idx)

                        # Generate pollution block
                        if Carbon_emission > self.env_recover_ability:
                            blanks = world.get_near_blank_grid(loc_r, loc_c)
                            if blanks:
                                pollute_r, pollute_c = random.choice(blanks)
                                world.maps.set_point("Carbon_pollution", pollute_r, pollute_c, 1)

                        # Receive payment for the house
                        income = self.payment * agent.state["Manufacture_volume"]
                        agent.state["inventory"]["Coin"] += income
                        assert income > 0, income

                        # Incur the Labor cost and Carbon_emission for building
                        agent.state["endogenous"]["Labor"] += self.labor * agent.state[
                            "Manufacture_volume"] if self.labor_multiple else self.labor
                        agent.state["endogenous"]["Carbon_emission"] += Carbon_emission

                        builds.append(
                            {
                                "enterprise": agent.idx,
                                "research_activity": research_activity,
                                "Carbon_emission_rate_change": previous_rate - agent.state["Carbon_emission_rate"],
                                "loc": np.array(agent.loc),
                                "Carbon_emission": Carbon_emission,
                                "Income": income,
                            }
                        )
                elif action == 2:
                    # Action Success
                    if random.random() > self.random_fails:
                        action_result = 'Success'
                        # Newest history add Research_mark
                        agent.state["Research_history"][0] = 1
                    else:
                        action_result = 'Fails'

                    research.append(
                        {
                            "enterprise": agent.idx,
                            "research_activity": research_activity,
                            "Carbon_emission_rate_change": previous_rate - agent.state["Carbon_emission_rate"],
                            "action_result": action_result,
                            "Research_count": agent.state["Research_count"][0]
                        }
                    )
                    agent.state["endogenous"]["Labor"] += self.labor * agent.state[
                        "Research_ability"] if self.labor_multiple else self.labor

                    agent.state["inventory"]["Coin"] -= self.payment/(2* agent.state["Research_ability"])

                else:
                    raise ValueError

        self.Manufactures["builds"].append(builds)
        self.Manufactures["research"].append(research)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "Research_ability": agent.state["Research_ability"],
                "Manufacture_volume": agent.state["Manufacture_volume"],
                "Carbon_emission_rate": agent.state["Carbon_emission_rate"],
                "Research_count": agent.state["Research_count"][0]
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([self.agent_can_build(agent), self.agent_can_research(agent)])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
        for actions in self.Manufactures["builds"]:
            for action in actions:
                idx = action["enterprise"]
                build_stats[idx]["n_builds"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in build_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        num_Property = np.sum(world.maps.get("Property") > 0)
        out_dict["total_builds"] = num_Property

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """
        world = self.world

        for agent in world.agents:
            if self.evaluate:
                assert self.n_agents == 5
                if agent.idx == 0:
                    agent.state["Research_ability"] = 1.2
                    agent.state["Manufacture_volume"] = 1.6
                elif agent.idx == 1:
                    agent.state["Research_ability"] = 1.0
                    agent.state["Manufacture_volume"] = 1.4
                elif agent.idx == 2:
                    agent.state["Research_ability"] = 1.2
                    agent.state["Manufacture_volume"] = 1.2
                elif agent.idx == 3:
                    agent.state["Research_ability"] = 1.4
                    agent.state["Manufacture_volume"] = 1.2
                elif agent.idx == 4:
                    agent.state["Research_ability"] = 1.6
                    agent.state["Manufacture_volume"] = 1.0
                else:
                    raise ValueError(agent.idx)
            else:
                # 0.5 ~ PMSM, 0.1.0.0
                agent.state["Research_ability"] = random.choice([1.0, 1.2, 1.4, 1.6])
                agent.state["Manufacture_volume"] = random.choice([1.0, 1.2, 1.4, 1.6])

            # initiate the Carbon_emission_rate be 1.0.0.0
            agent.state["Carbon_emission_rate"] = 1.0

            agent.state["Start_Er"] = 1.0

            # initiate the [total research count, this year research count] be [0, 0]
            agent.state["Research_count"] = [0, 0]

            agent.state["Research_history"] = [0] * max(self.delay, self.forget)

        self.Manufactures = {"builds": [], "research": []}

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.

        """
        return self.Manufactures
