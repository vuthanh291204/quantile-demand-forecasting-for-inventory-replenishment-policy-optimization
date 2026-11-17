"""Hand-calculated synthetic inventory cases, independent of M5 files."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from src.simulation.inventory import simulate_inventory


def test_scheduled_orders_match_hand_calculated_daily_ledger():
    result = simulate_inventory(
        [4, 5, 1, 8, 2],
        initial_stock=5,
        lead_time_days=2,
        order_quantities=[8, 0, 0, 4, 0],
        initial_orders={1: 2, 3: 6},
    )
    # day, opening, received, available, demand, filled, lost, closing,
    # ordered, on order, position. All rows were calculated by hand.
    expected = pd.DataFrame(
        [
            [1, 5, 2, 7, 4, 4, 0, 3, 8, 14, 17],
            [2, 3, 0, 3, 5, 3, 2, 0, 0, 14, 14],
            [3, 0, 14, 14, 1, 1, 0, 13, 0, 0, 13],
            [4, 13, 0, 13, 8, 8, 0, 5, 4, 4, 9],
            [5, 5, 0, 5, 2, 2, 0, 3, 0, 4, 7],
        ],
        columns=[
            "day", "opening_stock", "received", "available_stock", "demand",
            "fulfilled", "lost_sales", "closing_stock", "order_placed",
            "on_order", "inventory_position",
        ],
    )
    pd.testing.assert_frame_equal(result.daily, expected)
    assert result.pending_orders == {6: 4}


def test_lost_sales_are_not_carried_into_next_day():
    result = simulate_inventory(
        [5, 1], initial_stock=3, lead_time_days=1, order_quantities=[4, 0]
    )
    assert result.daily["fulfilled"].tolist() == [3, 1]
    assert result.daily["lost_sales"].tolist() == [2, 0]
    assert result.daily["closing_stock"].tolist() == [0, 3]
    assert result.daily["received"].tolist() == [0, 4]


def test_due_receipt_can_satisfy_first_day_demand():
    result = simulate_inventory(
        [4], initial_stock=0, lead_time_days=2, initial_orders={1: 5}
    )
    assert result.daily["fulfilled"].tolist() == [4]
    assert result.daily["closing_stock"].tolist() == [1]
    assert result.daily["on_order"].tolist() == [0]
    assert result.pending_orders == {}


def test_zero_demand_days_still_advance_delivery_time():
    result = simulate_inventory(
        [0, 0, 0, 2], initial_stock=0, lead_time_days=3,
        order_quantities=[3, 0, 0, 0],
    )
    assert result.daily["received"].tolist() == [0, 0, 0, 3]
    assert result.daily["closing_stock"].tolist() == [0, 0, 0, 1]
    assert result.pending_orders == {}


def test_orders_after_horizon_are_retained_with_absolute_day_numbers():
    result = simulate_inventory(
        [0, 0], initial_stock=0, lead_time_days=3, start_day=1606,
        order_quantities=[2, 4], initial_orders={1615: 7},
    )
    assert result.daily["day"].tolist() == [1606, 1607]
    assert result.daily["on_order"].tolist() == [9, 13]
    assert result.pending_orders == {1609: 2, 1610: 4, 1615: 7}


def test_omitted_orders_leave_empty_stock_and_record_all_demand_as_lost():
    result = simulate_inventory([0, 3, 1], initial_stock=0, lead_time_days=2)
    assert result.daily["fulfilled"].tolist() == [0, 0, 0]
    assert result.daily["lost_sales"].tolist() == [0, 3, 1]
    assert result.daily["order_placed"].tolist() == [0, 0, 0]
    assert result.pending_orders == {}


def test_policy_matches_hand_calculation_and_sees_state_before_new_order():
    seen = []

    def refill(state):
        seen.append(state)
        return max(10 - state.inventory_position, 0)

    result = simulate_inventory(
        [4, 5, 1, 8, 2], initial_stock=5, lead_time_days=2,
        initial_orders={1: 2, 3: 6}, policy=refill,
    )
    assert result.daily["order_placed"].tolist() == [1, 3, 1, 8, 2]
    assert result.daily["closing_stock"].tolist() == [3, 0, 6, 1, 0]
    assert result.daily["lost_sales"].tolist() == [0, 2, 0, 0, 0]
    assert result.daily["inventory_position"].tolist() == [10] * 5
    assert result.pending_orders == {6: 8, 7: 2}
    assert [state.inventory_position for state in seen] == [9, 7, 9, 2, 8]
    assert seen[0].on_hand == 3
    assert seen[0].on_order == 6
    assert seen[0].pending_orders == ((3, 6),)
    assert seen[0].lead_time_days == 2
    assert seen[0].demand_history == (4,)
    with pytest.raises(FrozenInstanceError):
        seen[0].on_hand = 99
    with pytest.raises(TypeError):
        seen[0].pending_orders[0] = (3, 99)


def test_future_demand_does_not_change_earlier_policy_observations_or_orders():
    def run(demand):
        seen = []

        def policy(state):
            seen.append(state)
            return sum(state.demand_history)

        result = simulate_inventory(
            demand, initial_stock=2, lead_time_days=2, policy=policy
        )
        return result, seen

    original, seen = run([2, 3, 1, 4])
    changed, changed_seen = run([2, 3, 99, 99])
    assert [state.demand_history for state in seen] == [
        (2,), (2, 3), (2, 3, 1), (2, 3, 1, 4),
    ]
    assert seen[:2] == changed_seen[:2]
    pd.testing.assert_frame_equal(original.daily.iloc[:2], changed.daily.iloc[:2])


def test_continuing_from_final_stock_and_pipeline_matches_single_run():
    whole = simulate_inventory(
        [4, 5, 1, 8, 2], initial_stock=5, lead_time_days=2,
        order_quantities=[8, 0, 0, 4, 0], initial_orders={1: 2, 3: 6},
    )
    first = simulate_inventory(
        [4, 5], initial_stock=5, lead_time_days=2,
        order_quantities=[8, 0], initial_orders={1: 2, 3: 6},
    )
    rest = simulate_inventory(
        [1, 8, 2], initial_stock=first.daily["closing_stock"].iloc[-1],
        lead_time_days=2, order_quantities=[0, 4, 0],
        initial_orders=first.pending_orders, start_day=3,
    )
    combined = pd.concat([first.daily, rest.daily], ignore_index=True)
    pd.testing.assert_frame_equal(combined, whole.daily)
    assert rest.pending_orders == whole.pending_orders


def test_numpy_integer_inputs_are_supported_and_left_untouched():
    demand = np.array([2, 5], dtype=np.int64)
    orders = np.array([3, 0], dtype=np.int64)
    initial_orders = {2: 1}
    result = simulate_inventory(
        demand, initial_stock=np.int64(3), lead_time_days=np.int64(1),
        order_quantities=orders, initial_orders=initial_orders,
    )
    assert result.daily["closing_stock"].tolist() == [1, 0]
    np.testing.assert_array_equal(demand, [2, 5])
    np.testing.assert_array_equal(orders, [3, 0])
    assert initial_orders == {2: 1}


@pytest.mark.parametrize(
    "overrides",
    [
        {"demand": []},
        {"demand": [-1]},
        {"demand": [1.5]},
        {"demand": [float("nan")]},
        {"demand": [float("inf")]},
        {"demand": [True]},
        {"initial_stock": -1},
        {"initial_stock": True},
        {"lead_time_days": 0},
        {"lead_time_days": 1.5},
        {"start_day": 0},
        {"order_quantities": [1, 2]},
        {"order_quantities": [-1]},
        {"initial_orders": {0: 2}},
        {"initial_orders": {2: -1}},
        {"initial_orders": {1.5: 2}},
        {"policy": 42},
        {"policy": lambda state: 0, "order_quantities": [0]},
    ],
)
def test_invalid_inputs_are_rejected(overrides):
    arguments = {"demand": [1], "initial_stock": 0, "lead_time_days": 1}
    arguments.update(overrides)
    with pytest.raises(ValueError):
        simulate_inventory(**arguments)


@pytest.mark.parametrize("quantity", [-1, 0.5, None, True, float("nan"), float("inf")])
def test_invalid_policy_quantities_are_rejected(quantity):
    with pytest.raises(ValueError, match="policy order on day 1"):
        simulate_inventory(
            [1], initial_stock=0, lead_time_days=1, policy=lambda state: quantity
        )
