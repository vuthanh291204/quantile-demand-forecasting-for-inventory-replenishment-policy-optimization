"""Daily single-item inventory simulation with lost sales.

Receive due orders at the start of the day, fulfil today's demand, record
lost sales, then place the day's replenishment at the end of the day.
An order placed on day t arrives at the start of day t + lead_time_days.
Quantities are whole units and each input entry represents one calendar day.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import pandas as pd

from src.simulation.policies import InventoryState, OrderingPolicy


@dataclass(frozen=True)
class SimulationResult:
    daily: pd.DataFrame
    pending_orders: dict[int, int]


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def simulate_inventory(
    demand: Sequence[int],
    *,
    initial_stock: int,
    lead_time_days: int,
    order_quantities: Sequence[int] | None = None,
    initial_orders: Mapping[int, int] | None = None,
    start_day: int = 1,
    policy: OrderingPolicy | None = None,
) -> SimulationResult:
    """Simulate one product at one location with a fixed delivery lead time.

    ``demand`` contains nonnegative integer quantities for consecutive days.
    ``order_quantities`` contains the quantities to order at each day's end.
    A supplied schedule is useful for checking the inventory arithmetic.
    Alternatively, ``policy`` receives an immutable end-of-day state and
    returns a nonnegative integer order quantity. Supply at most one of
    ``order_quantities`` and ``policy``. If both are omitted, orders are zero.

    ``initial_orders`` maps arrival day to the total quantity already due on
    that day. Arrival days use the same numbering as ``start_day``. Existing
    orders may arrive on the first simulated day. Orders that arrive after
    the last simulated day are returned in ``pending_orders``.

    ``initial_stock`` is the stock before the first day's receipts. Daily
    ``on_order`` and ``inventory_position`` are measured after placing that
    day's order. Unfulfilled demand is recorded as lost sales for that day.

    All inputs are supplied by the caller and are left untouched. Each day
    is processed, including days with zero demand. Operational parameter
    values are simulation assumptions supplied by the caller.
    """
    if policy is not None and order_quantities is not None:
        raise ValueError("supply either policy or order_quantities")
    if policy is not None and not callable(policy):
        raise ValueError("policy must be callable")
    stock = _integer(initial_stock, "initial_stock")
    lead_time = _integer(lead_time_days, "lead_time_days", minimum=1)
    first_day = _integer(start_day, "start_day", minimum=1)
    daily_demand = [_integer(value, "demand") for value in demand]
    if not daily_demand:
        raise ValueError("demand must contain at least one day")

    if order_quantities is None:
        scheduled_orders = [0] * len(daily_demand)
    else:
        scheduled_orders = [_integer(value, "order_quantities") for value in order_quantities]
    if len(scheduled_orders) != len(daily_demand):
        raise ValueError("order_quantities must have one quantity per demand day")

    pending = {}
    if initial_orders is not None:
        for arrival_day, quantity in initial_orders.items():
            arrival = _integer(arrival_day, "initial order arrival day", minimum=first_day)
            amount = _integer(quantity, "initial order quantity")
            if amount > 0:
                pending[arrival] = amount

    on_order = sum(pending.values())
    records = []
    for day, (requested, order) in enumerate(
        zip(daily_demand, scheduled_orders, strict=True), start=first_day
    ):
        opening_stock = stock
        received = pending.pop(day, 0)
        on_order -= received
        available_stock = opening_stock + received

        fulfilled = min(available_stock, requested)
        lost_sales = requested - fulfilled
        stock = available_stock - fulfilled

        if policy is not None:
            state = InventoryState(
                day=day,
                lead_time_days=lead_time,
                on_hand=stock,
                pending_orders=tuple(sorted(pending.items())),
                demand_history=tuple(daily_demand[: day - first_day + 1]),
            )
            order = _integer(policy(state), f"policy order on day {day}")

        if order > 0:
            arrival = day + lead_time
            pending[arrival] = pending.get(arrival, 0) + order
            on_order += order

        records.append(
            {
                "day": day,
                "opening_stock": opening_stock,
                "received": received,
                "available_stock": available_stock,
                "demand": requested,
                "fulfilled": fulfilled,
                "lost_sales": lost_sales,
                "closing_stock": stock,
                "order_placed": order,
                "on_order": on_order,
                "inventory_position": stock + on_order,
            }
        )

    return SimulationResult(
        daily=pd.DataFrame.from_records(records),
        pending_orders=dict(sorted(pending.items())),
    )


def main() -> None:
    """Print a synthetic example whose quantities can be checked by hand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-demo", action="store_true", help="use an illustrative ordering function"
    )
    args = parser.parse_args()

    def refill_to_ten(state: InventoryState) -> int:
        return max(10 - state.inventory_position, 0)

    result = simulate_inventory(
        demand=[4, 5, 1, 8, 2],
        initial_stock=5,
        lead_time_days=2,
        order_quantities=None if args.policy_demo else [8, 0, 0, 4, 0],
        initial_orders={1: 2, 3: 6},
        policy=refill_to_ten if args.policy_demo else None,
    )
    print("Synthetic example: all inputs are illustrative assumptions.")
    if args.policy_demo:
        print("Policy demo: order max(10 - inventory_position, 0) at each day's end.")
    print(result.daily.to_string(index=False))
    print(f"\nPending orders (arrival day -> quantity): {result.pending_orders}")


if __name__ == "__main__":
    main()
