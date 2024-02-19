import numpy as np


def coin_minus_labor(
        coin_endowment, total_labor, labor_coefficient
):
    # https://en.wikipedia.org/wiki/Isoelastic_utility

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = coin_endowment - util_l

    return util


def isoelastic_coin_minus_labor(
        coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):

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

    idx_used_mobile = np.exp(sum([-1 * mobile_coefficient * idx for idx in mobile_idx]))
    idx_used_planner = -10000 * int(remained_idx < 0)

    util = equality * prod * idx_used_mobile + idx_used_planner
    return util


def get_gini(endowments):

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

    return 1 - get_gini(endowments)


def get_productivity(coin_endowments):

    return np.sum(coin_endowments)


def planner_metrics(coin_endowments, mobile_idx, remained_idx, mobile_coefficient):
    n_agents = len(coin_endowments)
    prod = get_productivity(coin_endowments) / n_agents
    equality = get_equality(coin_endowments)

    idx_used_mobile = np.exp(sum([-1 * mobile_coefficient * idx for idx in mobile_idx]))
    idx_used_planner = -10000 * int(remained_idx < 0)

    util = equality * prod * idx_used_mobile + idx_used_planner

    planner_metrix = {
        "util": util,
        "equality": equality,
        "prod": prod,
        "mobile_idx_used": mobile_idx
    }
    return planner_metrix