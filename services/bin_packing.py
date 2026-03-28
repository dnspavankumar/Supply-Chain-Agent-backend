from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from models.schemas import Bin, BinPackingResult, Item, PositionedItem

EPSILON = 1e-6


@dataclass(frozen=True)
class _Orientation:
    length: float
    width: float
    height: float


@dataclass
class _Placement:
    item_id: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    weight_kg: float


@dataclass
class _FreeSpace:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float


def _item_volume(item: Item) -> float:
    return item.length * item.width * item.height


def _placement_volume(placement: _Placement) -> float:
    return placement.length * placement.width * placement.height


def _space_volume(space: _FreeSpace) -> float:
    return space.length * space.width * space.height


def _generate_orientations(item: Item) -> list[_Orientation]:
    orientations = {
        _Orientation(length=float(l), width=float(w), height=float(h))
        for l, w, h in permutations((item.length, item.width, item.height), 3)
    }
    return sorted(
        orientations,
        key=lambda orient: (
            -(orient.length * orient.width),
            -orient.height,
            -orient.length,
            -orient.width,
        ),
    )


def _sorted_items(items: list[Item]) -> list[Item]:
    return sorted(
        items,
        key=lambda item: (
            _item_volume(item),
            max(item.length, item.width, item.height),
            item.weight_kg,
        ),
        reverse=True,
    )


def _fits_bounds(
    x: float,
    y: float,
    z: float,
    orient: _Orientation,
    bin: Bin,
) -> bool:
    return (
        x + orient.length <= bin.length + EPSILON
        and y + orient.width <= bin.width + EPSILON
        and z + orient.height <= bin.height + EPSILON
    )


def _intersects(
    placement: _Placement,
    x: float,
    y: float,
    z: float,
    orient: _Orientation,
) -> bool:
    return not (
        x + orient.length <= placement.x + EPSILON
        or placement.x + placement.length <= x + EPSILON
        or y + orient.width <= placement.y + EPSILON
        or placement.y + placement.width <= y + EPSILON
        or z + orient.height <= placement.z + EPSILON
        or placement.z + placement.height <= z + EPSILON
    )


def _overlaps_any(
    placements: Iterable[_Placement],
    x: float,
    y: float,
    z: float,
    orient: _Orientation,
) -> bool:
    return any(_intersects(existing, x, y, z, orient) for existing in placements)


def _to_positioned_item(placement: _Placement) -> PositionedItem:
    return PositionedItem(
        item_id=placement.item_id,
        x=round(placement.x, 3),
        y=round(placement.y, 3),
        z=round(placement.z, 3),
        length=round(placement.length, 3),
        width=round(placement.width, 3),
        height=round(placement.height, 3),
    )


def _build_result(
    strategy_name: str,
    items: list[Item],
    placements: list[_Placement],
    bin: Bin,
) -> BinPackingResult:
    packed_ids = [placement.item_id for placement in placements]
    packed_lookup = set(packed_ids)
    unpacked_ids = [item.item_id for item in items if item.item_id not in packed_lookup]

    packed_volume = sum(_placement_volume(placement) for placement in placements)
    packed_weight = sum(placement.weight_kg for placement in placements)
    bin_volume = bin.length * bin.width * bin.height

    space_utilization = 0.0 if bin_volume <= EPSILON else (packed_volume / bin_volume) * 100.0
    weight_utilization = (packed_weight / bin.max_weight_kg) * 100.0

    return BinPackingResult(
        packed_items=packed_ids,
        unpacked_items=unpacked_ids,
        space_utilization_pct=round(max(0.0, min(space_utilization, 100.0)), 3),
        weight_utilization_pct=round(max(0.0, min(weight_utilization, 100.0)), 3),
        positions=[
            _to_positioned_item(placement).model_dump(mode="json")
            for placement in placements
        ],
        strategy=strategy_name,  # type: ignore[arg-type]
    )


def _split_guillotine(
    free_space: _FreeSpace,
    orient: _Orientation,
) -> list[_FreeSpace]:
    right = _FreeSpace(
        x=free_space.x + orient.length,
        y=free_space.y,
        z=free_space.z,
        length=free_space.length - orient.length,
        width=free_space.width,
        height=free_space.height,
    )
    front = _FreeSpace(
        x=free_space.x,
        y=free_space.y + orient.width,
        z=free_space.z,
        length=orient.length,
        width=free_space.width - orient.width,
        height=free_space.height,
    )
    top = _FreeSpace(
        x=free_space.x,
        y=free_space.y,
        z=free_space.z + orient.height,
        length=orient.length,
        width=orient.width,
        height=free_space.height - orient.height,
    )
    candidates = [right, front, top]
    return [
        space
        for space in candidates
        if space.length > EPSILON and space.width > EPSILON and space.height > EPSILON
    ]


def _space_contains(outer: _FreeSpace, inner: _FreeSpace) -> bool:
    return (
        outer.x <= inner.x + EPSILON
        and outer.y <= inner.y + EPSILON
        and outer.z <= inner.z + EPSILON
        and outer.x + outer.length >= inner.x + inner.length - EPSILON
        and outer.y + outer.width >= inner.y + inner.width - EPSILON
        and outer.z + outer.height >= inner.z + inner.height - EPSILON
    )


def _prune_free_spaces(free_spaces: list[_FreeSpace]) -> list[_FreeSpace]:
    cleaned = [
        space
        for space in free_spaces
        if space.length > EPSILON and space.width > EPSILON and space.height > EPSILON
    ]
    pruned: list[_FreeSpace] = []
    for index, space in enumerate(cleaned):
        if any(
            other_index != index and _space_contains(cleaned[other_index], space)
            for other_index in range(len(cleaned))
        ):
            continue
        pruned.append(space)
    return pruned


def guillotine_heuristic(items: list[Item], bin: Bin) -> BinPackingResult:
    placements: list[_Placement] = []
    used_weight = 0.0
    free_spaces = [
        _FreeSpace(
            x=0.0,
            y=0.0,
            z=0.0,
            length=bin.length,
            width=bin.width,
            height=bin.height,
        )
    ]

    for item in _sorted_items(items):
        if used_weight + item.weight_kg > bin.max_weight_kg + EPSILON:
            continue

        best_choice: tuple[tuple[float, float, float, float], int, _Orientation] | None = None
        for idx, free_space in enumerate(free_spaces):
            for orient in _generate_orientations(item):
                if (
                    orient.length > free_space.length + EPSILON
                    or orient.width > free_space.width + EPSILON
                    or orient.height > free_space.height + EPSILON
                ):
                    continue
                waste = _space_volume(free_space) - _item_volume(item)
                score = (waste, free_space.z, free_space.y, free_space.x)
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, idx, orient)

        if best_choice is None:
            continue

        _, free_index, orient = best_choice
        target = free_spaces.pop(free_index)
        placement = _Placement(
            item_id=item.item_id,
            x=target.x,
            y=target.y,
            z=target.z,
            length=orient.length,
            width=orient.width,
            height=orient.height,
            weight_kg=item.weight_kg,
        )
        placements.append(placement)
        used_weight += item.weight_kg

        free_spaces.extend(_split_guillotine(target, orient))
        free_spaces = _prune_free_spaces(free_spaces)

    return _build_result("guillotine_heuristic", items, placements, bin)


def _is_point_inside_placement(point: tuple[float, float, float], placement: _Placement) -> bool:
    x, y, z = point
    return (
        placement.x + EPSILON < x < placement.x + placement.length - EPSILON
        and placement.y + EPSILON < y < placement.y + placement.width - EPSILON
        and placement.z + EPSILON < z < placement.z + placement.height - EPSILON
    )


def _normalize_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(point[0], 6), round(point[1], 6), round(point[2], 6))


def _contact_score(
    x: float,
    y: float,
    z: float,
    orient: _Orientation,
    placements: list[_Placement],
) -> float:
    score = 0.0

    if z <= EPSILON:
        score += orient.length * orient.width

    if x <= EPSILON:
        score += orient.width * orient.height
    if y <= EPSILON:
        score += orient.length * orient.height

    for placement in placements:
        if abs((placement.z + placement.height) - z) <= EPSILON:
            overlap_x = min(x + orient.length, placement.x + placement.length) - max(x, placement.x)
            overlap_y = min(y + orient.width, placement.y + placement.width) - max(y, placement.y)
            if overlap_x > EPSILON and overlap_y > EPSILON:
                score += overlap_x * overlap_y

    return score


def _clean_points(
    points: set[tuple[float, float, float]],
    placements: list[_Placement],
    bin: Bin,
) -> set[tuple[float, float, float]]:
    cleaned: set[tuple[float, float, float]] = set()
    for point in points:
        x, y, z = point
        if x < -EPSILON or y < -EPSILON or z < -EPSILON:
            continue
        if x > bin.length + EPSILON or y > bin.width + EPSILON or z > bin.height + EPSILON:
            continue
        if any(_is_point_inside_placement(point, placement) for placement in placements):
            continue
        cleaned.add(_normalize_point(point))
    return cleaned


def extreme_point_rule(items: list[Item], bin: Bin) -> BinPackingResult:
    placements: list[_Placement] = []
    used_weight = 0.0
    points: set[tuple[float, float, float]] = {(0.0, 0.0, 0.0)}

    for item in _sorted_items(items):
        best_choice: tuple[
            tuple[float, float, float, float, float],
            tuple[float, float, float],
            _Orientation,
        ] | None = None

        for point in sorted(points, key=lambda p: (p[2], p[1], p[0])):
            x, y, z = point
            for orient in _generate_orientations(item):
                if used_weight + item.weight_kg > bin.max_weight_kg + EPSILON:
                    continue
                if not _fits_bounds(x, y, z, orient, bin):
                    continue
                if _overlaps_any(placements, x, y, z, orient):
                    continue

                support = _contact_score(x, y, z, orient, placements)
                residual = (
                    (bin.length - (x + orient.length))
                    + (bin.width - (y + orient.width))
                    + (bin.height - (z + orient.height))
                )
                score = (z, y, x, -support, residual)
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, point, orient)

        if best_choice is None:
            continue

        _, point, orient = best_choice
        x, y, z = point
        placement = _Placement(
            item_id=item.item_id,
            x=x,
            y=y,
            z=z,
            length=orient.length,
            width=orient.width,
            height=orient.height,
            weight_kg=item.weight_kg,
        )
        placements.append(placement)
        used_weight += item.weight_kg

        new_points = {
            (x + orient.length, y, z),
            (x, y + orient.width, z),
            (x, y, z + orient.height),
        }
        points.discard(point)
        points.update(new_points)
        points = _clean_points(points, placements, bin)

    return _build_result("extreme_point_rule", items, placements, bin)


def deepest_bottom_left(items: list[Item], bin: Bin) -> BinPackingResult:
    placements: list[_Placement] = []
    used_weight = 0.0
    candidate_points: set[tuple[float, float, float]] = {(0.0, 0.0, 0.0)}

    for item in _sorted_items(items):
        best_choice: tuple[
            tuple[float, float, float, float, float],
            tuple[float, float, float],
            _Orientation,
        ] | None = None

        for point in sorted(candidate_points, key=lambda p: (p[2], p[1], -p[0])):
            x, y, z = point
            for orient in _generate_orientations(item):
                if used_weight + item.weight_kg > bin.max_weight_kg + EPSILON:
                    continue
                if not _fits_bounds(x, y, z, orient, bin):
                    continue
                if _overlaps_any(placements, x, y, z, orient):
                    continue

                residual = (
                    abs(bin.length - (x + orient.length))
                    + abs(bin.width - (y + orient.width))
                    + abs(bin.height - (z + orient.height))
                )
                score = (z, y, -x, residual, -_contact_score(x, y, z, orient, placements))
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, point, orient)

        if best_choice is None:
            continue

        _, point, orient = best_choice
        x, y, z = point
        placement = _Placement(
            item_id=item.item_id,
            x=x,
            y=y,
            z=z,
            length=orient.length,
            width=orient.width,
            height=orient.height,
            weight_kg=item.weight_kg,
        )
        placements.append(placement)
        used_weight += item.weight_kg

        candidate_points.discard(point)
        candidate_points.update(
            {
                (x + orient.length, y, z),
                (x, y + orient.width, z),
                (x, y, z + orient.height),
            }
        )
        candidate_points = _clean_points(candidate_points, placements, bin)

    return _build_result("deepest_bottom_left", items, placements, bin)


def best_bin_packing_result(items: list[Item], bin: Bin) -> BinPackingResult:
    results = [
        guillotine_heuristic(items, bin),
        extreme_point_rule(items, bin),
        deepest_bottom_left(items, bin),
    ]
    return max(
        results,
        key=lambda result: (
            result.space_utilization_pct,
            result.weight_utilization_pct,
            len(result.packed_items),
        ),
    )
