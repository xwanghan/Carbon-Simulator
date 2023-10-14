# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from Carbon_simulator.foundation import utils
from Carbon_simulator.foundation.agents import agent_registry as agents
from Carbon_simulator.foundation.components import component_registry as components
from Carbon_simulator.foundation.entities import endogenous_registry as endogenous
from Carbon_simulator.foundation.entities import landmark_registry as landmarks
from Carbon_simulator.foundation.entities import resource_registry as resources
from Carbon_simulator.foundation.scenarios import scenario_registry as scenarios


def make_env_instance(scenario_name, **kwargs):
    scenario_class = scenarios.get(scenario_name)
    return scenario_class(**kwargs)
