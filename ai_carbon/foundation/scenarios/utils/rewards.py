# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np


def coin_minus_labor(
        coin_endowment, total_labor, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1.0.0. 0 yields utility
            that increases linearly with coin. 1.0.0 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = coin_endowment - util_l

    return util


def isoelastic_coin_minus_labor(
        coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1.0.0. 0 yields utility
            that increases linearly with coin. 1.0.0 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, coin_endowment))
    else:  # isoelastic_eta >= 0
        if np.all(coin_endowment >= 0):
            util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)
        else:
            util_c = coin_endowment - 1

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def planner_strategy(coin_endowments, mobile_idx, remained_idx, mobile_coefficient):
    n_agents = len(coin_endowments)
    prod = get_productivity(coin_endowments) / n_agents
    equality = get_equality(coin_endowments)

    idx_used_mobile = sum([-1 * mobile_coefficient * idx for idx in mobile_idx])
    idx_used_planner = -10000 * int(remained_idx < 0)

    util = equality * prod + idx_used_mobile + idx_used_planner
    return util


def get_gini(endowments):
    """Returns the normalized Gini index describing the distribution of endowments.

    https://en.wikipedia.org/wiki/Gini_coefficient

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized Gini index for the distribution of endowments (float). A value of 1
            indicates everything belongs to 1 agent (perfect inequality), whereas a
            value of 0 indicates all agents have equal endowments (perfect equality).

    Note:
        Uses a slightly different method depending on the number of agents. For fewer
        agents (<30), uses an exact but slow method. Switches to using a much faster
        method for more agents, where both methods produce approximately equivalent
        results.
    """
    n_agents = len(endowments)

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = np.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = np.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = np.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * np.sum(
        np.cumsum(s_endows) / (np.sum(s_endows) + 1e-10)
    )


def get_equality(endowments):
    """Returns the complement of the normalized Gini index (equality = 1 - Gini).

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized equality index for the distribution of endowments (float). A value
            of 0 indicates everything belongs to 1 agent (perfect inequality),
            whereas a value of 1 indicates all agents have equal endowments (perfect
            equality).
    """
    return 1 - get_gini(endowments)


def get_productivity(coin_endowments):
    """Returns the total coin inside the simulated economy.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Total coin endowment (float).
    """
    return np.sum(coin_endowments)


def planner_metrics(coin_endowments, mobile_idx, remained_idx, mobile_coefficient):
    n_agents = len(coin_endowments)
    prod = get_productivity(coin_endowments) / n_agents
    equality = get_equality(coin_endowments)

    idx_used_mobile = sum([-1 * mobile_coefficient * idx for idx in mobile_idx])
    idx_used_planner = -10000 * int(remained_idx < 0)

    util = equality * prod + idx_used_mobile + idx_used_planner

    planner_metrix = {
        "util": util,
        "equality": equality,
        "prod": prod,
        "mobile_idx_used": mobile_idx
    }
    return planner_metrix