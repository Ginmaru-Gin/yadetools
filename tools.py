from __future__ import annotations
import numpy as np
from typing import Callable
from yade import O, utils, plot
from yade.utils import Vector3


DEBUG = True


sphere_Vr = lambda r: 4 / 3 * np.pi * r**3
sphere_Vd = lambda d: sphere_Vr(0.5 * d)


def static_state_checker(
    *,
    ids: list[int] | None = None,
    static_time: float,
    static_degree: float = 0.05,
    timeout: float = np.inf,
) -> tuple[Callable, Callable]:
    """Returns callable object and attributes updater for that object"""

    if ids is None:
        ids = [id for id in range(len(O.bodies)) if O.bodies[id] is not None]
    else:
        ids = [id for id in ids if O.bodies[id] is not None]

    allowed_displacement = (
        np.sum(
            [
                O.bodies[id].shape.radius
                for id in ids
                if isinstance(O.bodies[id].shape, utils.Sphere)
            ]
        )
        * static_degree
    )

    prev_positions = np.array(((np.inf, np.inf, np.inf),) * len(ids))
    time_creating = O.time
    last_check_time = O.time

    if DEBUG:
        print(
            f"{staticStateChecker} log:\n"
            f"\t{len(ids)=}\n"
            f"\t{allowed_displacement=}\n"
            f"\t{static_time=}\n"
            f"\t{timeout=}"
        )

    def check() -> bool:
        """Checks is the scene balanced or not"""

        nonlocal prev_positions
        nonlocal last_check_time

        # TODO rework timeout, it's wrong
        if O.time - time_creating > timeout:
            print(f"staticChecker: timeout ({O.time})")
            return True

        if O.time - last_check_time < static_time:
            return False

        positions = np.array([O.bodies[id].state.pos for id in ids])
        disp_vectors = map(utils.Vector3, np.abs(positions - prev_positions))
        displacement = sum(v.norm() for v in disp_vectors)
        if displacement <= allowed_displacement:
            return True
        last_check_time = O.time
        prev_positions = positions
        if DEBUG:
            print(f"{displacement=}")

    def update(
        *,
        new_ids: list[int] | None = None,
        new_static_time: float | None = None,
        new_static_degree: float | None = None,
        new_timeout: float | None = None,
    ):
        nonlocal allowed_displacement
        nonlocal ids
        nonlocal prev_positions
        nonlocal static_time
        nonlocal timeout

        if new_ids is not None:
            ids = new_ids
            prev_positions = np.array(((np.inf, np.inf, np.inf),) * len(ids))
        if new_static_time is not None:
            static_time = new_static_time
        if new_static_degree is not None:
            allowed_displacement = (
                np.sum(
                    [
                        O.bodies[id].shape.radius
                        for id in ids
                        if isinstance(O.bodies[id].shape, utils.Sphere)
                    ]
                )
                * new_static_degree
            )
        if new_timeout is not None:
            timeout = new_timeout

    return check, update


def staticStateChecker(
    *,
    ids: list[int] | None = None,
    static_time: float,
    static_degree: float | None = None,
    timeout: float = np.inf,
) -> Callable[[], bool]:
    """
    DEPRECATED
    Returns callable object that checks is the scene balanced or not
    """

    if ids is None:
        ids = [id for id in range(len(O.bodies)) if O.bodies[id] is not None]
    else:
        ids = [id for id in ids if O.bodies[id] is not None]
    if static_degree is None:
        static_degree = 0.05

    allowed_displacement = (
        np.sum(
            [
                O.bodies[id].shape.radius
                for id in ids
                if isinstance(O.bodies[id].shape, utils.Sphere)
            ]
        )
        * static_degree
    )

    prev_positions = np.array(((np.inf, np.inf, np.inf),) * len(ids))
    timeout += O.time
    timing = O.time

    if DEBUG:
        print(
            f"{staticStateChecker} log:\n"
            f"\t{len(ids)=}\n"
            f"\t{allowed_displacement=}\n"
            f"\t{static_time=}\n"
            f"\t{timeout=}"
        )

    def checker() -> bool:
        """Checks is the scene balanced or not"""

        nonlocal prev_positions
        nonlocal timing

        # TODO rework timeout, it's wrong
        if O.time > timeout:
            print(f"staticChecker: timeout ({O.time})")
            return True

        if O.time - timing < static_time:
            return False

        positions = np.array([O.bodies[id].state.pos for id in ids])
        disp_vectors = map(utils.Vector3, np.abs(positions - prev_positions))
        displacement = sum(v.norm() for v in disp_vectors)
        if displacement <= allowed_displacement:
            return True
        timing = O.time
        prev_positions = positions
        if DEBUG:
            print(f"{displacement=}")

    return checker


def get_contact_area(ids: list[int]):
    intrs = np.hstack([O.bodies[id].intrs() for id in ids])
    sphere_ids = [
        i.id2
        for i in intrs
        if O.bodies[i.id2] is not None
        and isinstance(O.bodies[i.id2].shape, utils.Sphere)
    ]
    contact_area = np.sum([np.pi * O.bodies[id].shape.radius ** 2 for id in sphere_ids])
    return contact_area


def _in_bounds(body, bounds):
    return bool(np.prod([bounds[0][i] < body.state.pos[i] < bounds[1][i] for i in range(3)]))


def filter_in_bounds(bounds, ids):
    return [id for id in ids if _in_bounds(O.bodies[id], bounds)]


def filterInBounds(bounds, ids):
    """
    DEPRECATED, replacement: filter_in_bounds
    """
    return filter_in_bounds(bounds, ids)


def filter_not_in_bounds(bounds, ids):
    return [id for id in ids if not _in_bounds(O.bodies[id], bounds)]


def erase(ids):
    for id in ids:
        O.bodies.erase(id)
    return ids


def _get_interactions_by_ids(ids):
    return [(i.id1, i.id2) for i in O.interactions if i.isReal and i.id1 in ids]


def get_porosity(ids, bounds):
    filtered_ids = filter_in_bounds(bounds, ids)
    interactions = _get_interactions_by_ids(filtered_ids)
    volTotal = (Vector3(Vector3(bounds[1]) - Vector3(bounds[0]))).prod()
    volSphs = sum(4 / 3 * np.pi * O.bodies[id].shape.radius ** 3 for id in filtered_ids)
    ds = [
        (
            2 * O.bodies[id1].shape.radius
            - O.interactions[id1, id2].geom.penetrationDepth,
            O.bodies[id1].shape.radius,
        )
        for id1, id2 in interactions
    ]
    volContacts = sum([1 / 12 * np.pi * (4 * r + d) * (2 * r - d) ** 2 for d, r in ds])
    volSolid = volSphs - volContacts
    volVoid = volTotal - volSolid
    porosity = volVoid / volTotal
    porosity_coeff = porosity / (1 - porosity) if porosity != 1 else np.inf
    return porosity_coeff
