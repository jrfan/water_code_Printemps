import numpy as np

def custom_quota(
    self,
    crisis_level: int,
    actors_priority: np.ndarray,
    avg_pump: np.ndarray,
    DOE: float,
    DCR: float,
) -> np.ndarray:
    priority_weights_1=0.5984838332004384
    priority_weights_2=0.2726664411087506
    priority_weights_3=1.4211296405877707
    compression_1=0.7021381013959085
    compression_2=0.1439255494194097
    compression_3=0.3844465144976456
    compression_4=0.4488361312435196
    priority_weights = (priority_weights_1, priority_weights_1+priority_weights_2, priority_weights_1+priority_weights_2+priority_weights_3)
    compression = (compression_1+compression_2+compression_3+compression_4,
                   compression_1+compression_2+compression_3,
                   compression_1+compression_2,
                   compression_1)
    weights = np.select(
        [actors_priority == 0, actors_priority == 1, actors_priority == 2],
        priority_weights
    )
    weighted_demand = avg_pump * weights
    total_weighted = np.sum(weighted_demand)

    total_available = np.sum(avg_pump) * compression[crisis_level] * 0.9
    quota = total_available * (weighted_demand / total_weighted)

    return np.clip(quota, 0.0, 1.2 * avg_pump)


def custom_incentive_policy(
    self,
    actions: np.ndarray,
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE: float = 15,
    DCR: float = 10,
) -> np.ndarray:
    eco_alpha=2.951164369798566
    eco_beta=1.243923687349587
    fine_scale=1.007582757371356
    overuse_power=2.2350381182010137
    subsidy_low=0.7391821888277132
    penalty_low=1.1335955057300822
    penalty_medium=1.0022170966918416
    reward_onquota_low=0.008427389904452721
    reward_onquota_med=0.007170277608364876
    reward_onquota_high=0.029970840681204116
    fine_cap_factor=2.2064066712213544
    reward_onquota_med = reward_onquota_low+reward_onquota_med
    reward_onquota_high = reward_onquota_med+reward_onquota_high
    n = len(avg_incomes)
    incentives = np.zeros(n)

    current_flow = water_flows[-1]
    crisis_level = is_crisis[-1]
    eco_pressure = 1.0 + eco_alpha * max(0.0, (DOE - current_flow) / max(DOE - DCR, 1e-6)) ** eco_beta

    priority_penalty = np.select(
        [actors_priority == 0, actors_priority == 1, actors_priority == 2],
        [penalty_low, penalty_medium, 1.0]
    )
    reward_onquota = np.select(
        [actors_priority == 0, actors_priority == 1, actors_priority == 2],
        [reward_onquota_low, reward_onquota_med, reward_onquota_high]
    )

    for i in range(n):
        overuse = max(0.0, water_pump[i] - quota[i])
        if overuse > 0:
            penalty = fine_scale * (overuse ** overuse_power) / max(avg_pump[i], 1e-3)
            penalty *= eco_pressure * priority_penalty[i]

            if actors_priority[i] == 0:
                penalty = min(penalty, fine_cap_factor * max(avg_incomes[i], 1.0))

            incentives[i] = penalty

        elif water_pump[i] <= quota[i] * 0.9 and actors_priority[i] >= 1:
            incentives[i] = -reward_onquota[i] * max(avg_incomes[i], 0.0)

        elif crisis_level >= 1 and actors_priority[i] == 0:
            incentives[i] = -subsidy_low * max(avg_incomes[i], 0.0)

    return incentives
