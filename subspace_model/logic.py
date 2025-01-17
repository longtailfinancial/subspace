from math import ceil, floor
from random import randint
from typing import Callable

from cadCAD.types import PolicyOutput, StateUpdateFunction  # type: ignore

from subspace_model.const import *
from subspace_model.metrics import *
from subspace_model.types import *


def generic_policy(_1, _2, _3, _4) -> dict:
    """Function to generate pass through policy

    Args:
        _1
        _2
        _3
        _4

    Returns:
        dict: Empty dictionary
    """
    return {}


def replace_suf(variable: str, default_value=0.0) -> Callable:
    """Creates replacing function for state update from string

    Args:
        variable (str): The variable name that is updated

    Returns:
        function: A function that continues the state across a substep
    """
    return lambda _1, _2, _3, state, signal: (
        variable,
        signal.get(variable, default_value),
    )


def add_suf(variable: str, default_value=0.0) -> Callable:
    """Creates replacing function for state update from string

    Args:
        variable (str): The variable name that is updated

    Returns:
        function: A function that continues the state across a substep
    """
    return lambda _1, _2, _3, state, signal: (
        variable,
        signal.get(variable, default_value) + state[variable],
    )


## Time Tracking ##


def p_evolve_time(params: SubspaceModelParams, _2, _3, _4) -> PolicyOutput:
    """ """
    delta_days = params["timestep_in_days"]
    delta_seconds = delta_days * DAY_TO_SECONDS
    delta_blocks = delta_seconds / params["block_time_in_seconds"]
    return {
        "delta_days": delta_days,
        "days_passed": delta_days,
        "delta_blocks": delta_blocks,
        "blocks_passed": delta_blocks,
    }


## Farmer Rewards ##


def p_fund_reward(_1, _2, _3, state: SubspaceModelState) -> PolicyOutput:
    """
    Farmer rewards that originates from the DSF.
    """
    dsf_share = state["dsf_relative_disbursal_per_day"] ** state["delta_days"]
    reward = state["fund_balance"] * dsf_share
    return {"block_reward": reward, "fund_balance": -reward}


def p_issuance_reward(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    Farmer rewards that originates from protocol issuance.
    XXX: there's a hard cap on how much can be issued.
    """
    issuance_per_day = params["issuance_function"](params, state)
    reward = issuance_per_day * state["delta_days"]

    # Make sure that the protocol has tokens to issue
    if reward > state["reward_issuance_balance"]:
        reward = state["reward_issuance_balance"]
    else:
        pass

    return {"block_reward": reward, "reward_issuance_balance": -reward}


def p_split_reward(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """ """
    reward = state["block_reward"]
    reward_to_fund = (
        reward * params["reward_proposer_share"] * params["fund_tax_on_proposer_reward"]
    )
    reward_to_farmers = reward - reward_to_fund
    return {"farmers_balance": reward_to_farmers, "fund_balance": reward_to_fund}


## Operator Rewards


def p_operator_reward(_1, _2, _3, _4) -> PolicyOutput:
    """
    Protocol issued rewards to Staked Operators.
    XXX: Assumed to be zero
    """
    reward = 0.0
    return {"other_issuance_balance": -reward, "operators_balance": reward}


## Environmental processes


def s_average_priority_fee(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the ts-average priority fee during an timestep through
    a Gaussian process.
    XXX: depends on an stochastic process assumption.
    """
    return (
        "average_priority_fee",
        max(
            params["priority_fee_function"](params, state),
            0,
        ),
    )


def s_average_compute_weight_per_tx(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the ts-average compute weights per transaction through a Gaussian process.
    XXX: depends on an stochastic process assumption.
    """
    return (
        "average_compute_weight_per_tx",
        max(
            params["compute_weights_per_tx_function"](params, state),
            params["min_compute_weights_per_tx"],
        ),
    )


def s_average_compute_weight_per_bundle(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the ts-average compute weights per transaction through a Gaussian process.
    XXX: depends on an stochastic process assumption.
    """
    # TODO: verify that is implemented correctly
    return (
        "average_compute_weight_per_budle",
        max(
            params["compute_weight_per_bundle_function"](
                params,
                state,
            ),
            params["min_compute_weights_per_bundle"],
        ),
    )


def s_average_compute_weight_per_bundle(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the ts-average compute weights per transaction through a Gaussian process.
    XXX: depends on an stochastic process assumption.
    """
    # TODO: verify that is implemented correctly
    return (
        "average_compute_weight_per_budle",
        max(
            params["compute_weight_per_bundle_function"](
                params,
                state,
            ),
            params["min_compute_weights_per_bundle"],
        ),
    )


def s_average_transaction_size(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the ts-average transaction size through a Gaussian process.
    XXX: depends on an stochastic process assumption.
    """
    return (
        "average_transaction_size",
        max(
            params["transaction_size_function"](
                params,
                state,
            ),
            params["min_transaction_size"],
        ),
    )


def s_transaction_count(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the ts-average transaction size through a Poisson process.
    XXX: depends on an stochastic process assumption.
    """
    transaction_count = max(
        params["transaction_count_per_day_function"](
            params,
            state,
        ),
        0,
    )

    return ("transaction_count", transaction_count)


def s_bundle_count(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """
    Simulate the bs-average transaction size through a Poisson process.
    XXX: depends on an stochastic process assumption.
    """
    # TODO: refactor
    bundle_count = max(
        params["bundle_count_per_day_function"](
            params,
            state,
        ),
        0,
    )

    return ("bundle_count", bundle_count)


def p_archive(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    TODO: check if underlying assumptions / terminology are valid.
    TODO: revisit assumption on the supply & demand matching.
    FIXME: homogenize terminology
    """
    # The header volume. The bytes stored in the chain for each transaction header.
    header_volume: Bytes = state["delta_blocks"] * params["header_size"]

    # Transaction volume in bytes. This is the driver of blockchain history storage growth.
    tx_volume: Bytes = state["transaction_count"] * state["average_transaction_size"]

    # New buffer bytes are transaction volume plus header
    new_buffer_bytes: Bytes = tx_volume + header_volume

    # The new size of the buffer
    current_buffer: Bytes = new_buffer_bytes + state["buffer_size"]

    # Number of segments needed for the current buffer
    segments_being_archived: int = int(
        floor(current_buffer / params["archival_buffer_segment_size"])
    )

    # Remove the segments from the buffer and place them in the history
    new_history_bytes: Bytes = 0
    if segments_being_archived > 0:
        new_buffer_bytes += -1 * SEGMENT_SIZE * segments_being_archived
        new_history_bytes += SEGMENT_HISTORY_SIZE * segments_being_archived

    # Update the blockchain history size and the current buffer size
    return {
        "blockchain_history_size": new_history_bytes,
        "buffer_size": new_buffer_bytes,
    }


def p_pledge_sectors(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    Decide amount of pledged bytes to be added.

    XXX: depends on an stochastic process assumption.
    XXX: emits a minimum required space to keep  up with the blockchain history size.
    """

    # Size of history in bytes
    blockchain_history_size: Bytes = state["blockchain_history_size"]

    # Minimum replication factor
    min_replication_factor: float = params["min_replication_factor"]

    # Previous total_space_pledged
    total_space_pledged: Bytes = state["total_space_pledged"]

    # Required total space to be pledged
    required_space_pledged: Bytes = blockchain_history_size * min_replication_factor

    # Required new space to be pledged
    new_pledge_due_to_requirements: Bytes = max(
        required_space_pledged - total_space_pledged, 0
    )

    # New pledge random parameter function
    new_pledge_due_to_random: Bytes = (
        int(
            max(
                params["new_sectors_per_day_function"](params, state),
                0,
            )
        )
        * SECTOR_SIZE
    )

    # Add the random process amount to the minimum amount.
    # Take at least the minimum.
    new_space_pledged = max(
        new_pledge_due_to_requirements + new_pledge_due_to_random,
        new_pledge_due_to_requirements,
    )

    # Update the total space pledged.
    return {"total_space_pledged": new_space_pledged}


## Compute & Operator Fees
def p_storage_fees(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    Calculate storage fees.

    If holders balance is insufficient, then the amount of paid fees
    will be lower even though the transactions still go through.

    References:
    - https://github.com/subspace/subspace/blob/53dca169379e65b4fb97b5c7753f5d00bded2ef2/crates/pallet-transaction-fees/src/lib.rs#L271
    """
    # Input
    total_credit_supply: Credits = params["credit_supply_definition"](state)
    total_space_pledged: Bytes = state["total_space_pledged"]
    blockchain_history_size: Bytes = state["blockchain_history_size"]
    min_replication_factor: float = params["min_replication_factor"]

    # Add bundle storage
    average_bundle_size: Bytes = max(
        params["bundle_size_function"](params, state), params["min_bundle_size"]
    )
    bundle_storage: Bytes = average_bundle_size * state[
        "bundle_count_per_day_function"
    ](params, state)

    if total_space_pledged <= blockchain_history_size * min_replication_factor:
        raise ValueError(
            "total_space_pledged <= blockchain_history_size * min_replication_factor"
        )

    free_space: Bytes = max(
        (total_space_pledged / min_replication_factor) - blockchain_history_size, 1
    )

    storage_fee_in_credits_per_bytes = (
        total_credit_supply / free_space
    )  # Credits / Bytes

    extrinsic_length_in_bytes: Bytes = (
        state["transaction_count"] * state["average_transaction_size"]
    )

    # storage_fee(tx) as per spec
    storage_fee_volume: Credits = (
        storage_fee_in_credits_per_bytes * extrinsic_length_in_bytes
    )

    # HACK : Constrain total_storage_fees to 1/2 all holders balance
    # TODO : Add comment as to why this is needed.
    eff_storage_fee_volume: Credits = min(
        storage_fee_volume, state["holders_balance"] / 2
    )

    # Storage Fees
    # Fee distribution
    storage_fees_to_fund: Credits = (
        params["fund_tax_on_storage_fees"] * eff_storage_fee_volume
    )
    storage_fees_to_farmers: Credits = eff_storage_fee_volume - storage_fees_to_fund

    return {
        # Fee Calculation
        "free_space": free_space,
        "storage_fee_in_credits_per_bytes": storage_fee_in_credits_per_bytes,
        "extrinsic_length_in_bytes": extrinsic_length_in_bytes,
        "storage_fee_volume": eff_storage_fee_volume,
        # Reward Distribution
        "storage_fees_to_farmers": storage_fees_to_farmers,
        "farmers_balance": storage_fees_to_farmers,
        "storage_fees_to_fund": storage_fees_to_fund,
        "fund_balance": storage_fees_to_fund,
        "holders_balance": -eff_storage_fee_volume,
    }


def p_compute_fees(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    Calculate compute fees.

    If holders balance is insufficient, then the amount of paid fees
    will be lower even though the transactions still go through.

    Reference: https://subspacelabs.notion.site/Fees-Rewards-Specification-WIP-1b835c7684a940f188920802ca6791f2#4d2c4f4b69a94fcca49a7fcaca7563cc
    """

    weight_to_fee: Credits = params["weight_to_fee"]
    max_normal_weight: Picoseconds = 0.75 * BLOCK_WEIGHT_FOR_2_SEC
    max_bundle_weight = state["max_bundle_weight"]  # TODO: type
    target_block_fullness = state["target_block_fullness"]  # TODO: type
    block_weight_utilization = state["block_utilization"]  # TODO: type
    adjustment_variable = state["adjustment_variable"]  # TODO: type
    priority_fee_volume = state["average_priority_fee"]  # TODO: type

    target_block_delta = target_block_fullness - block_weight_utilization  # TODO: type

    targeted_adjustment_parameter = (  # TODO: type
        1
        + adjustment_variable * target_block_delta
        + adjustment_variable**2 * target_block_delta**2 / 2
    )

    prev_compute_fee_multiplier = state["compute_fee_multiplier"]  # TODO: type
    compute_fee_multiplier = (
        targeted_adjustment_parameter * prev_compute_fee_multiplier
    )  # TODO: type

    # Caculate compute weight
    tx_compute_weight: ComputeWeights = (
        state["average_compute_weight_per_tx"] * state["transaction_count"]
    )
    bundles_compute_weight: ComputeWeights = (
        state["average_compute_weight_per_bundle"] * state["bundle_count"]
    )

    total_compute_weights: ComputeWeights = tx_compute_weight + bundles_compute_weight
    eff_minimum_fee: Credits = 1 * SHANNON_IN_CREDITS

    # Calculate compute fee volume
    compute_fee_volume: Credits = max(
        (
            compute_fee_multiplier * weight_to_fee * total_compute_weights
            + priority_fee_volume
        ),
        eff_minimum_fee,
    )

    # Constrain compute fee volume to be less than holders balance
    eff_compute_fee_volume: Credits = min(compute_fee_volume, state["holders_balance"])
    eff_scale: float = eff_compute_fee_volume / compute_fee_volume

    fees_to_distribute: Credits = compute_fee_volume

    # Bundle relevant fees go to operators rather than farmers
    bundle_share_of_weight = bundles_compute_weight / total_compute_weights

    # Fee volume to be from bundles
    fees_from_bundles: Credits = fees_to_distribute * bundle_share_of_weight

    # Fee volume for farmers
    fees_to_farmers: Credits = fees_to_distribute - fees_from_bundles

    # # Calculate the fees to operators
    fees_to_operators: Credits = fees_from_bundles

    # Calculate total fees
    total_fees: Credits = fees_to_farmers + fees_to_operators

    # TODO: check if fees goes to the farmers/nominators balance
    # or if is auto-staked

    return {
        # Compute fee calculations
        "target_block_delta": target_block_delta,
        "targeted_adjustment_parameter": targeted_adjustment_parameter,
        "compute_fee_multiplier": compute_fee_multiplier,
        "tx_compute_weight": tx_compute_weight,
        "compute_fee_volume": eff_compute_fee_volume,
        "priority_fee_volume": priority_fee_volume,
        # Taking and distributing fees
        "farmers_balance": fees_to_farmers,
        "operators_balance": fees_to_operators,
        "holders_balance": -eff_compute_fee_volume,
        "fees_to_operators": fees_to_operators,
    }


def p_slash(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    XXX: depends on an stochastic process assumption.
    TODO: validate if correct
    """
    slash_value = 0.0
    slash_to_fund = 0.0
    slash_to_holders = 0.0
    operator_shares_to_subtract = 0.0
    slash_to_burn = 0.0

    # XXX: no slash occurs if the pool balance is zero.
    pool_balance = state["staking_pool_balance"]
    if pool_balance > 0:
        slash_count = params["slash_per_day_function"](
            params,
            state,
        )
        slash_value = min(
            slash_count * params["slash_function"](params, state), pool_balance
        )
        if slash_value > 0:
            slash_to_fund = slash_value * params["slash_to_fund"]
            slash_to_holders = slash_value * params["slash_to_holders"]
            slash_to_burn = slash_value - (slash_to_fund + slash_to_holders)

            # XXX: we assume that the slash is aplied on the staking pool
            # and that its effect is to reduce the operator shares
            # by using an invariant product as a assumption.

            pool_balance_after = pool_balance - slash_value
            total_shares = (
                state["operator_pool_shares"] + state["nominator_pool_shares"]
            )
            operator_shares_to_subtract = total_shares * (
                pool_balance_after / pool_balance - 1.0
            )
        else:
            slash_value = 0.0

    return {
        "staking_pool_balance": -slash_value,
        "fund_balance": slash_to_fund,
        "holders_balance": slash_to_holders,
        "operator_pool_shares": operator_shares_to_subtract,
        "burnt_balance": slash_to_burn,
    }


def p_unvest(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    Impl notes: 30% of total.
    22% to be unvested with 24mo and 8% to be unvested with 48mo.
    25% total to be unlocked after 12mo and linearly afterwards.

    # TODO: parametrize / generalize the schedule
    # TODO: what happens if there's less than 51% community owned?
    """
    if state["days_passed"] < 365:
        allocated_tokens = 0.0
    elif state["days_passed"] >= 365:
        allocated_tokens = 0.30 * 0.25
        allocated_tokens += (
            0.22 * 0.75 * min((state["days_passed"] - 365) / (365 * 2), 1)
        )  # Investors
        allocated_tokens += (
            0.08 * 0.75 * min((state["days_passed"] - 365) / (4 * 365), 1)
        )  # Team
        allocated_tokens *= params["max_credit_supply"]

    tokens_to_allocate = allocated_tokens - state["allocated_tokens"]
    holders_balance = tokens_to_allocate
    other_issuance_balance = -holders_balance

    return {
        "other_issuance_balance": other_issuance_balance,
        "holders_balance": holders_balance,
        "allocated_tokens": allocated_tokens,
    }


### User Behavioral Processes


def p_staking(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    XXX: this assumes that operators and nominators will always
    stake a given % of their free balance every timestep.
    XXX: assumes an invariant product
    TODO: enforce minimum staking amounts
    """
    if state["operator_pool_shares"] > 0 or state["nominator_pool_shares"] > 0:
        invariant = state["staking_pool_balance"] / (
            state["operator_pool_shares"] + state["nominator_pool_shares"]
        )
        # invariant = 1
    elif state["operator_pool_shares"] == 0 and state["nominator_pool_shares"] == 0:
        invariant = 1
    else:
        invariant = None

    # Stake operation
    operator_stake_fraction = params["operator_stake_per_ts_function"](
        params,
        state,
    )

    if operator_stake_fraction > 0:
        operator_stake = state["operators_balance"] * operator_stake_fraction
    elif invariant > 0:
        operator_stake = (
            state["operator_pool_shares"] * operator_stake_fraction * invariant
        )
    else:
        operator_stake = 0.0

    nominator_stake_fraction = params["nominator_stake_per_ts_function"](
        params,
        state,
    )

    if nominator_stake_fraction > 0:
        nominator_stake = state["nominators_balance"] * nominator_stake_fraction
    elif invariant > 0:
        nominator_stake = (
            state["nominator_pool_shares"] * nominator_stake_fraction * invariant
        )
    else:
        nominator_stake = 0.0

    total_stake = operator_stake + nominator_stake

    # NOTE: for handling withdraws bigger than the pool itself.
    if -total_stake > state["staking_pool_balance"]:
        old_total_stake = total_stake
        total_stake = -state["staking_pool_balance"]
        scale = total_stake / old_total_stake
        operator_stake *= scale
        nominator_stake *= scale

    return {
        "operators_balance": -operator_stake,
        "operator_pool_shares": operator_stake / invariant,
        "nominator_pool_shares": nominator_stake / invariant,
        "nominators_balance": -nominator_stake,
        "staking_pool_balance": total_stake,
    }


def p_transfers(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState
) -> PolicyOutput:
    """
    XXX: stakeholders will always transfer a give % of their balance every ts
    """
    delta_nominators = 0.0
    delta_holders = 0.0
    delta_farmers = 0.0
    delta_operators = 0.0

    # Farmers to Holders
    if state["farmers_balance"] > 0:
        delta = state["farmers_balance"] * params[
            "transfer_farmer_to_holder_per_day_function"
        ](params, state)
        delta_farmers -= delta
        delta_holders += delta

    # Operators to Holders
    if state["operators_balance"] > 0:
        delta = state["operators_balance"] * params[
            "transfer_operator_to_holder_per_day_function"
        ](params, state)
        delta_operators -= delta
        delta_holders += delta

    # Holder to Nominators
    if state["holders_balance"] > 0:
        delta = state["holders_balance"] * params[
            "transfer_holder_to_nominator_per_day_function"
        ](params, state)
        delta_holders -= delta
        delta_nominators += delta

        # Holder to Operators
        delta = state["holders_balance"] * params[
            "transfer_holder_to_operator_per_day_function"
        ](params, state)
        delta_holders -= delta
        delta_operators += delta

    return {
        "operators_balance": delta_operators,
        "holders_balance": delta_holders,
        "nominators_balance": delta_nominators,
        "farmers_balance": delta_farmers,
    }


def s_block_utilization(
    params: SubspaceModelParams, _2, _3, state: SubspaceModelState, _5
) -> StateUpdateFunction:
    """ """
    size = state["transaction_count"] * state["average_transaction_size"]
    max_size = (
        params["max_block_size"] * DAY_TO_SECONDS * params["block_time_in_seconds"]
    )
    value = size / max_size
    return ("block_utilization", value)


def p_reference_subsidy(
    params: SubspaceModelParams, _2, state_history: list, state: SubspaceModelState
) -> PolicyOutput:
    """ """
    previous_block_time = state_history[-1][-1]["blocks_passed"]
    current_block_time = state["blocks_passed"]
    rewards_during_last_timestep = 0.0
    for t in range(int(previous_block_time), int(current_block_time)):
        rewards_during_last_timestep += sum(
            [component(t) for component in params["reference_subsidy_components"]]
        )
    return {
        "reference_subsidy": rewards_during_last_timestep,
    }
