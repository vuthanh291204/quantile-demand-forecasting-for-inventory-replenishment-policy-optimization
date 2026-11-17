"""Ordering policy interface for the daily inventory simulator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class InventoryState:
    """End-of-day observations before placing the current day's order.

    ``pending_orders`` contains (arrival_day, quantity) pairs in date order.
    ``demand_history`` contains observed demand from the simulation start
    through the current day, inclusive. Both collections are immutable.
    """

    day: int
    lead_time_days: int
    on_hand: int
    pending_orders: tuple[tuple[int, int], ...]
    demand_history: tuple[int, ...]

    @property
    def on_order(self) -> int:
        return sum(quantity for _, quantity in self.pending_orders)

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


OrderingPolicy = Callable[[InventoryState], int]
