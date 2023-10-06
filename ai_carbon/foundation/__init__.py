# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_carbon.foundation import utils
from ai_carbon.foundation.agents import agent_registry as agents
from ai_carbon.foundation.components import component_registry as components
from ai_carbon.foundation.entities import endogenous_registry as endogenous
from ai_carbon.foundation.entities import landmark_registry as landmarks
from ai_carbon.foundation.entities import resource_registry as resources
from ai_carbon.foundation.scenarios import scenario_registry as scenarios


def make_env_instance(scenario_name, **kwargs):
    scenario_class = scenarios.get(scenario_name)
    return scenario_class(**kwargs)
