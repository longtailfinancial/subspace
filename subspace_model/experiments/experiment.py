from copy import deepcopy

import pandas as pd

# import panel as pn
from cadCAD_tools import easy_run  # type: ignore
from pandas import DataFrame

from subspace_model.const import *
from subspace_model.experiments.logic import (
    DEFAULT_ISSUANCE_FUNCTION,
    ISSUANCE_FUNCTION,
)
from subspace_model.params import BASE_PARAMS, INITIAL_STATE, ISSUANCE_FOR_FARMERS
from subspace_model.structure import SUBSPACE_MODEL_BLOCKS
from subspace_model.types import SubspaceModelParams


# @pn.cache(to_disk=True)
def sanity_check_run(
    SIMULATION_DAYS: int = 700, TIMESTEP_IN_DAYS: int = 1, SAMPLES: int = 1
) -> DataFrame:
    """This function runs the model with simplified base parameters.

    Returns:
        DataFrame: A dataframe of simulation data
    """
    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    # print("BASE_PARAMS['deterministic']=", BASE_PARAMS["deterministic"])

    # %%
    # Get the sweep params in the form of single length arrays
    sweep_params = {k: [v] for k, v in BASE_PARAMS.items()}

    print(sweep_params)
    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)
    return sim_df


# @pn.cache(to_disk=True)
def standard_stochastic_run(
    SIMULATION_DAYS: int = 700, TIMESTEP_IN_DAYS: int = 1, SAMPLES: int = 30
) -> DataFrame:
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """
    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    # %%
    # Get the sweep params in the form of single length arrays
    sweep_params = {k: [v] for k, v in BASE_PARAMS.items()}

    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)
    return sim_df


# @pn.cache(to_disk=True)
def issuance_sweep(
    SIMULATION_DAYS: int = 700, TIMESTEP_IN_DAYS: int = 1, SAMPLES: int = 15
) -> DataFrame:
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """

    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    # %%
    # Get the sweep params in the form of single length arrays

    param_set_1 = BASE_PARAMS
    param_set_2 = deepcopy(BASE_PARAMS)
    param_set_2['label'] = 'alternate-issuance-function'
    param_set_2['issuance_function'] = (
        lambda *args, **kwargs: ISSUANCE_FOR_FARMERS / 5 * 365
    )

    param_sets = [param_set_1, param_set_2]

    sweep_params: dict[str, list] = {k: [] for k in BASE_PARAMS.keys()}
    for param_set in param_sets:
        for k, v in param_set.items():
            sweep_params[k].append(v)

    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)
    return sim_df


# @pn.cache(to_disk=True)
def fund_inclusion(
    SIMULATION_DAYS: int = 700, TIMESTEP_IN_DAYS: int = 1, SAMPLES: int = 15
) -> DataFrame:
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """
    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    FUND_TAX_ON_PROPOSER_REWARD = 0
    FUND_TAX_ON_STORAGE_FEES = 0
    SLASH_TO_FUND = 0

    # Get the sweep parameters in the form of single length arrays
    param_set_1 = BASE_PARAMS
    param_set_2 = deepcopy(BASE_PARAMS)
    param_set_2['label'] = 'no-fund'
    param_set_2['fund_tax_on_proposer_reward'] = FUND_TAX_ON_PROPOSER_REWARD
    param_set_2['fund_tax_on_storage_fees'] = FUND_TAX_ON_STORAGE_FEES
    param_set_2['slash_to_fund'] = SLASH_TO_FUND
    param_sets = [param_set_1, param_set_2]

    # Create the sweep parameters dictionary
    sweep_params: dict[str, list] = {k: [] for k in BASE_PARAMS.keys()}
    for param_set in param_sets:
        for k, v in param_set.items():
            sweep_params[k].append(v)

    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)

    # Return the simulation results dataframe
    return sim_df


# @pn.cache(to_disk=True)
def sanity_check_deterministic_run(
    SIMULATION_DAYS: int = 700,
    TIMESTEP_IN_DAYS: int = 1,
    SAMPLES: int = 1,
) -> DataFrame:
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """
    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    BASE_PARAMS['deterministic'] = True
    # print("BASE_PARAMS['deterministic']=", BASE_PARAMS["deterministic"])

    # %%
    # Get the sweep params in the form of single length arrays
    sweep_params = {k: [v] for k, v in BASE_PARAMS.items()}

    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)
    return sim_df


# @pn.cache(to_disk=True)
def reward_split_sweep(
    SIMULATION_DAYS: int = 700,
    TIMESTEP_IN_DAYS: int = 1,
    SAMPLES: int = 15,
) -> DataFrame:
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """
    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    # %%
    # Get the sweep params in the form of single length arrays

    param_set_1 = BASE_PARAMS
    param_set_2 = deepcopy(BASE_PARAMS)
    param_set_2['label'] = 'alternate-split'
    param_set_2['reward_proposer_share'] = 0.5

    param_sets = [param_set_1, param_set_2]

    sweep_params: dict[str, list] = {k: [] for k in BASE_PARAMS.keys()}
    for param_set in param_sets:
        for k, v in param_set.items():
            sweep_params[k].append(v)

    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)
    return sim_df


# @pn.cache(to_disk=True)
def standard_issuance_sweep(
    SIMULATION_DAYS: int = 700, TIMESTEP_IN_DAYS: int = 1, SAMPLES: int = 15
) -> DataFrame:
    """Sweeps issuance functions.

    Returns:
        DataFrame: A dataframe of simulation data
    """
    TIMESTEPS = int(SIMULATION_DAYS / TIMESTEP_IN_DAYS) + 1

    # %%
    # Get the sweep params in the form of single length arrays

    param_set_1 = BASE_PARAMS
    param_set_2 = deepcopy(BASE_PARAMS)
    param_set_2['label'] = 'standard-issuance-sweep'
    param_set_2['issuance_function'] = DEFAULT_ISSUANCE_FUNCTION

    param_sets = [param_set_1, param_set_2]

    sweep_params: dict[str, list] = {k: [] for k in BASE_PARAMS.keys()}
    for param_set in param_sets:
        for k, v in param_set.items():
            sweep_params[k].append(v)

    # Load simulation arguments
    sim_args = (INITIAL_STATE, sweep_params, SUBSPACE_MODEL_BLOCKS, TIMESTEPS, SAMPLES)

    # Run simulation
    sim_df = easy_run(*sim_args)
    return sim_df
