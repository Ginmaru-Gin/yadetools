from __future__ import annotations
import numpy as np
from typing import Callable
from yade import O, utils


DEBUG = False


sphere_Vr = lambda r: 4 / 3 * np.pi * r**3
sphere_Vd = lambda d: sphere_Vr(0.5 * d)


def staticStateChecker(
    *,
    ids: list[int] | None = None,
    static_time: float,
    static_degree: float | None = None,
    timeout: float = np.inf,
) -> Callable[[], bool]:
    """Returns callable object that checks is the scene balanced or not"""

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
    timing = O.time + static_time

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

        positions = np.array([O.bodies[id].state.pos for id in ids])
        displacement = np.sum(np.abs(positions - prev_positions))
        if displacement > allowed_displacement:
            timing = O.time
        prev_positions = positions
        if DEBUG:
            print(f"{displacement=}")
        return O.time - timing > static_time

    return checker


def get_contact_area(ids: list[int]):
    intrs = np.hstack([O.bodies[id].intrs() for id in ids])
    sphere_ids = [
        i.id2
        for i in intrs
        if O.bodies[i.id2] is not None and isinstance(O.bodies[i.id2].shape, utils.Sphere)
    ]
    contact_area = np.sum([np.pi * O.bodies[id].shape.radius ** 2 for id in sphere_ids])
    return contact_area
