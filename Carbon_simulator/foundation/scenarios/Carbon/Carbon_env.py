from copy import deepcopy

import numpy as np
from scipy import signal

from Carbon_simulator.foundation.base.base_env import BaseEnvironment, scenario_registry
from Carbon_simulator.foundation.scenarios.utils import rewards


@scenario_registry.add
class Carbon_env(BaseEnvironment):

    name = "Carbon/Carbon_env"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["Carbon_idx", "Carbon_emission", "Coin", "Property", "Carbon_pollution", "Labor", "Carbon_project", "Green_project"]

    def __init__(
            self,
            *base_env_args,
            planner_gets_spatial_info=True,
            full_observability=False,
            mobile_agent_observation_range=5,

            starting_agent_coin=0,
            isoelastic_eta=0.23,
            energy_cost=0.21,
            energy_warmup_constant=10000,
            energy_warmup_method="auto",

            mobile_coefficient=0.2,

            **base_env_kwargs
    ):
        super().__init__(*base_env_args, **base_env_kwargs)

        # Whether agents receive spatial information in their observation tensor
        self._planner_gets_spatial_info = bool(planner_gets_spatial_info)

        # Whether the (non-planner) agents can see the whole world map
        self._full_observability = bool(full_observability)

        self._mobile_agent_observation_range = int(mobile_agent_observation_range)

        # How much coin do agents begin with at upon reset
        self.starting_agent_coin = float(starting_agent_coin)
        assert self.starting_agent_coin >= 0.0

        self.isoelastic_eta = float(isoelastic_eta)
        assert 0.0 <= self.isoelastic_eta <= 1.0

        self.energy_cost = float(energy_cost)
        assert self.energy_cost >= 0

        self.energy_warmup_method = energy_warmup_method.lower()
        assert self.energy_warmup_method in ["decay", "auto"]
        # Decay constant for annealing to full energy cost
        # (if energy_warmup_constant == 0, there is no annealing)
        self.energy_warmup_constant = float(energy_warmup_constant)
        assert self.energy_warmup_constant >= 0
        self._auto_warmup_integrator = 0

        self.mobile_coefficient = mobile_coefficient

        # Use this to calculate marginal changes and deliver that as reward
        self.init_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.prev_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.curr_optimization_metric = {agent.idx: 0 for agent in self.all_agents}

    @property
    def energy_weight(self):
        """
        Energy annealing progress. Multiply with self.energy_cost to get the
        effective energy coefficient.
        """
        if self.energy_warmup_constant <= 0.0:
            return 1.0

        if self.energy_warmup_method == "decay":
            raise NotImplementedError

        if self.energy_warmup_method == "auto":
            return float(
                1.0
                - np.exp(-self._auto_warmup_integrator / self.energy_warmup_constant)
            )

        raise NotImplementedError

    def get_current_optimization_metrics(self):
        """
        Compute optimization metrics based on the current state. Used to compute reward.

        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        curr_optimization_metric = {}
        # (for agents)
        for agent in self.world.agents:
            curr_optimization_metric[agent.idx] = rewards.isoelastic_coin_minus_labor(
                coin_endowment=agent.total_endowment("Coin"),
                total_labor=agent.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
            )
        # (for the planner)
        curr_optimization_metric[self.world.planner.idx] = rewards.planner_strategy(
            coin_endowments=np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            ),
            mobile_idx=self.world.planner.state["settlement_idx"],
            remained_idx=self.world.planner.state["remained_idx"],
            mobile_coefficient=self.mobile_coefficient
        )

        return curr_optimization_metric

    def make_source_prob_maps(self):
        """
        Make maps specifying how likely each location is to be assigned as a resource
        source tile.

        Returns:
            source_prob_maps (dict): Contains a source probability map for both
                stone and wood
        """
        prob_gradient = (
                np.arange(self.world_size[0])[:, None].repeat(self.world_size[1], axis=1)
                ** self.gradient_steepness
        )
        prob_gradient = prob_gradient / np.mean(prob_gradient)

        return {
            "Carbon_project": prob_gradient * self.layout_specs["Carbon_project"]["starting_coverage"],
        }

    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------

    def reset_starting_layout(self):
        """
        Part 1.0.0/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).

        Here, generate a resource source layout consistent with target parameters.
        """
        self.world.maps.clear()

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).

        Here, empty inventories, give mobile agents any starting coin, and place them
        in random accessible locations to start.
        """
        self.world.clear_agent_locs()

        for agent in self.world.agents:
            # Clear everything to start with
            agent.state["inventory"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["escrow"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["endogenous"] = {k: 0 for k in agent.endogenous.keys()}
            # Add starting coin
            agent.state["inventory"]["Coin"] = float(self.starting_agent_coin)

        # Clear everything for the planner
        self.world.planner.state["inventory"] = {
            k: 0 for k in self.world.planner.inventory.keys()
        }
        self.world.planner.state["escrow"] = {
            k: 0 for k in self.world.planner.escrow.keys()
        }

        # Place the agents randomly in the world
        for agent in self.world.get_random_order_agents():
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(0, self.world_size[0])
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
            self.world.set_agent_loc(agent, r, c)

    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.

        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.

        In this class of scenarios, the scenario step handles stochastic resource
        regeneration.
        """

    def generate_observations(self):
        """
        Generate observations associated with this scenario.

        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.

        Here, non-planner agents receive spatial observations (depending on the env
        config) as well as the contents of their inventory and endogenous quantities.
        The planner also receives spatial observations (again, depending on the env
        config) as well as the inventory of each of the mobile agents.
        """
        obs = {}
        curr_map = self.world.maps.state

        owner_map = self.world.maps.owner_state
        loc_map = self.world.loc_map
        agent_idx_maps = np.concatenate([owner_map, loc_map[None, :, :]], axis=0)
        agent_idx_maps += 2
        agent_idx_maps[agent_idx_maps == 1] = 0

        agent_locs = {
            str(agent.idx): {
                "loc-row": agent.loc[0] / self.world_size[0],
                "loc-col": agent.loc[1] / self.world_size[1],
            }
            for agent in self.world.agents
        }
        agent_invs = {
            str(agent.idx): {
                "inventory-" + k: v * self.inv_scale for k, v in agent.inventory.items()
            }
            for agent in self.world.agents
        }

        obs[self.world.planner.idx] = {
            "inventory-" + k: v * self.inv_scale
            for k, v in self.world.planner.inventory.items()
        }
        if self._planner_gets_spatial_info:
            obs[self.world.planner.idx].update(
                dict(map=curr_map.transpose(1, 2, 0), idx_map=agent_idx_maps.transpose(1, 2, 0))
            )

        # Mobile agents see the full map. Convey location info via one-hot map channels.
        if self._full_observability:
            for agent in self.world.agents:
                my_map = np.array(agent_idx_maps)
                my_map[my_map == int(agent.idx) + 2] = 1
                sidx = str(agent.idx)
                obs[sidx] = {"map": curr_map, "idx_map": my_map}
                obs[sidx].update(agent_invs[sidx])

        # Mobile agents only see within a window around their position
        else:
            w = (
                self._mobile_agent_observation_range
            )  # View halfwidth (only applicable without full observability)

            padded_map = np.pad(
                curr_map,
                [(0, 1), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 1), (0, 0), (0, 0)],
            )

            padded_idx = np.pad(
                agent_idx_maps,
                [(0, 0), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 0), (0, 0), (0, 0)],
            )

            for agent in self.world.agents:
                r, c = [c + w for c in agent.loc]
                visible_map = padded_map[
                              :, (r - w): (r + w + 1), (c - w): (c + w + 1)
                              ]
                visible_idx = np.array(
                    padded_idx[:, (r - w): (r + w + 1), (c - w): (c + w + 1)]
                )

                visible_idx[visible_idx == int(agent.idx) + 2] = 1

                sidx = str(agent.idx)

                obs[sidx] = {"map": visible_map.transpose(1, 2, 0), "idx_map": visible_idx.transpose(1, 2, 0)}
                obs[sidx].update(agent_locs[sidx])
                obs[sidx].update(agent_invs[sidx])

                # Agent-wise planner info (gets crunched into the planner obs in the
                # base scenario code)
                obs["p" + sidx] = agent_invs[sidx]
                if self._planner_gets_spatial_info:
                    obs["p" + sidx].update(agent_locs[sidx])

        return obs

    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.

        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.

        Rewards are computed as the marginal utility (agents) or marginal social
        welfare (planner) experienced on this timestep. Ignoring discounting,
        this means that agents' (planner's) objective is to maximize the utility
        (social welfare) associated with the terminal state of the episode.
        """

        # "curr_optimization_metric" hasn't been updated yet, so it gives us the
        # utility from the last step.
        utility_at_end_of_last_time_step = deepcopy(self.curr_optimization_metric)

        # compute current objectives and store the values
        self.curr_optimization_metric = self.get_current_optimization_metrics()

        # reward = curr - prev objectives
        rew = {
            k: float(v - utility_at_end_of_last_time_step[k])
            for k, v in self.curr_optimization_metric.items()
        }

        # store the previous objective values
        self.prev_optimization_metric.update(utility_at_end_of_last_time_step)

        avg_agent_rew = np.mean([rew[a.idx] for a in self.world.agents])

        if avg_agent_rew > 0:
            self._auto_warmup_integrator += 1

        return rew

    # Optional methods for customization
    # ----------------------------------

    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.

        For this scenario, this method resets optimization metric trackers.
        """
        # compute current objectives
        curr_optimization_metric = self.get_current_optimization_metrics()

        self.curr_optimization_metric = deepcopy(curr_optimization_metric)
        self.init_optimization_metric = deepcopy(curr_optimization_metric)
        self.prev_optimization_metric = deepcopy(curr_optimization_metric)

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).

        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)

        Here, summarize social metrics, endowments, utilities, and labor cost annealing
        """
        metrics = dict()

        curr_optimization_metric = {}
        # (for agents)
        for agent in self.world.agents:
            curr_optimization_metric[agent.idx] = rewards.isoelastic_coin_minus_labor(
                coin_endowment=agent.total_endowment("Coin"),
                total_labor=agent.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
            )
        # (for the planner)
        curr_optimization_metric[self.world.planner.idx] = rewards.planner_metrics(
            coin_endowments=np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            ),
            mobile_idx=self.world.planner.state["settlement_idx"],
            remained_idx=self.world.planner.state["remained_idx"],
            mobile_coefficient=self.mobile_coefficient
        )

        for agent in self.all_agents:
            for resource, quantity in agent.inventory.items():
                metrics[
                    "endow/{}/{}".format(agent.idx, resource)
                ] = agent.total_endowment(resource)

            if agent.endogenous is not None:
                for resource, quantity in agent.endogenous.items():
                    metrics["endogenous/{}/{}".format(agent.idx, resource)] = quantity

            metrics["util/{}".format(agent.idx)] = curr_optimization_metric[
                agent.idx
            ]

        # Labor weight
        metrics["labor/weighted_cost"] = self.energy_cost * self.energy_weight
        metrics["labor/warmup_integrator"] = int(self._auto_warmup_integrator)

        return metrics
