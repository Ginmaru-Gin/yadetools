from __future__ import annotations
from collections.abc import Iterable
from copy import copy
import itertools
from random import random, shuffle
import numpy as np
from yade import O, pack, Vector3

from .tools import sphere_Vd


DEBUG = False


class Mixture:
    """Represents mixture of several spheres packages

    May be 'virtual' or 'non-virtual'.
    If object is 'virtual' it can be used for generating and adding  spheres to a scene.
    If object is 'non-virtual' it can be used to get lists of spheres indices on a scene.

    Properties:
    all_ids -- list of all spheres indices (only for 'non-virtual')
    ids     -- dictionary that represents lists of spheres indices by packages (only for 'non-virtual')
    """

    _packs: list[tuple[pack.SpherePack, str, tuple[float, float, float], str]]
    _ids: dict[str : set[int]]
    _virtual: bool

    def __init__(self, ids: dict[str : Iterable[int]] = dict()) -> None:
        self._packs = list()
        self._virtual = not bool(ids)
        self._ids = dict()
        for k, v in ids.items():
            self._ids[k] = set(v)

    @property
    def ids(self):
        """Dictionary that represents lists of spheres indices by packages"""

        if self._virtual:
            raise ValueError("Allowed only for non-virtual mixtures.")
        return self._ids

    @property
    def all_ids(self):
        """List of all spheres indices"""

        if self._virtual:
            raise ValueError("Allowed only for non-virtual mixtures.")
        return list(itertools.chain(*self._ids.values()))

    @property
    def virtual(self):
        """Status of object"""

        return self._virtual

    @property
    def aabb(self) -> tuple[Vector3, Vector3]:
        """Geometric bounds of object"""

        min_corner = Vector3.Zero
        max_corner = Vector3.Zero
        axis = [0, 1, 2]
        for ax in axis:
            if self._virtual:
                for p, _, _, _ in self._packs:
                    min_corner[ax] = np.min(
                        [min_corner[ax], np.min([s[0][ax] + s[1] for s in p])]
                    )
                    max_corner[ax] = np.max(
                        [max_corner[ax], np.max([s[0][ax] + s[1] for s in p])]
                    )
            else:
                min_corner[ax] = np.min(
                    [
                        O.bodies[id].state.pos[ax] - O.bodies[id].shape.radius
                        for id in self.all_ids
                    ]
                )
                max_corner[ax] = np.max(
                    [
                        O.bodies[id].state.pos[ax] + O.bodies[id].shape.radius
                        for id in self.all_ids
                    ]
                )

        return (min_corner, max_corner)
    
    def __add__(self, other: Mixture) -> Mixture:
        if self._virtual or other._virtual:
            raise ValueError("Allowed only for non-virtual mixtures.")

        mix = Mixture()
        for obj in [self, other]:
            for k, v in obj._ids.items():
                if not k in mix._ids:
                    mix._ids[k] = set()
                mix._ids[k].update(v)
        mix._virtual = False
        return mix

    def generate(
        self,
        min_corner: Vector3,
        max_corner: Vector3,
        *,
        num: int | None = None,
        grow_axis: str | None = None,
        labels: list[str],
        parts: list[list[float] | None],
        psd_sizes: list[list[float] | float],
        mix_parts: list[float],
        material_labels: list[str],
        distribute_mass: bool,
        colors: list[tuple[float, float, float]] | None = None,
        seed: int = -1,
        disp: float = 0,
    ) -> None:
        """TODO"""

        if not num is None:
            if grow_axis is None:
                raise ValueError(
                    "'grow_axis' argument is necessary with 'num' argument."
                )
            grow_axis = grow_axis.strip()
            if grow_axis not in {"x", "y", "z"}:
                raise ValueError("Acceptable growAxis values: 'x', 'y' or 'z'.")
        if colors is None:
            colors = np.repeat(None, len(parts))
        if not (
            len(parts)
            == len(psd_sizes)
            == len(material_labels)
            == len(mix_parts)
            == len(colors)
            == len(labels)
        ):
            raise ValueError(
                "Dimensions of 'parts', 'psdSizes', 'mix_parts', 'materials', 'labels' and 'colors' must be equals."
            )

        min_corner = Vector3(min_corner)
        max_corner = Vector3(max_corner)

        grow_ind = {"x": 0, "y": 1, "z": 2}[grow_axis.lower()]
        work_indices = list(range(3))
        work_indices.remove(grow_ind)

        Mv_list = []
        for part, psd_size in zip(parts, psd_sizes):
            if part is None:
                Mv = sphere_Vd(psd_size)
            else:
                part = list(copy(part))
                psd_size = list(copy(psd_size))
                part.insert(0, 0)
                psd_size.insert(0, 0)
                part = np.diff(part)
                volumes = [sphere_Vd(d) for d in psd_size]
                mean_volumes = [
                    np.mean((i, j)) for i, j in zip(volumes[:-1], volumes[1:])
                ]
                Mv = np.sum([v * p for v, p in zip(mean_volumes, part)])

            Mv_list.append(Mv)

        max_diameter = np.max(np.hstack(psd_sizes))
        cell_size = max_diameter * (1 + 2 * disp)
        mean_volume = np.mean(Mv_list)

        dims = np.floor((max_corner - min_corner) / cell_size)
        if num is None:
            num = int(dims.prod())
        else:
            dims[grow_ind] = np.ceil(num / np.prod([dims[i] for i in work_indices]))
        dims = list(map(int, dims))
        max_corner[grow_ind] = min_corner[grow_ind] + dims[grow_ind] * cell_size

        init_coord = min_corner + Vector3(np.repeat(cell_size * 0.5, 3))
        coords = [
            init_coord + Vector3(x, y, z) * cell_size
            for x, y, z in itertools.product(*[range(dims[i]) for i in range(3)])
        ]
        coords = [
            c + (1 if random() > 0.5 else -1) * (max_diameter * disp * random())
            for c in coords
        ]
        shuffle(coords)

        mix_parts = [p * mean_volume / mv for p, mv in zip(mix_parts, Mv_list)]
        sum_parts = np.sum(mix_parts)
        mix_parts = [p * (1 / sum_parts) for p in mix_parts]
        mix_nums = [int(np.floor(i * num)) for i in mix_parts]

        centers_list = np.split(coords, np.cumsum(mix_nums))

        sp = pack.SpherePack()
        radiuses_list = []
        for part, psd_size, mix_num in zip(parts, psd_sizes, mix_nums):
            if part is None:
                l = np.repeat(0.5 * psd_size, mix_num).tolist()
            else:
                sp.makeCloud(
                    minCorner=min_corner,
                    maxCorner=max_corner,
                    psdSizes=psd_size,
                    psdCumm=part,
                    num=mix_num,
                    distributeMass=distribute_mass,
                    seed=seed,
                )
                l = [s[1] for s in sp.toList()]

            radiuses_list.append(l)

        for radiuses, centers, material, color, label in zip(
            radiuses_list, centers_list, material_labels, colors, labels
        ):
            spheres = [(c, r) for c, r in zip(centers, radiuses)]
            p = pack.SpherePack()
            p.fromList(spheres)
            self._packs.append((p, material, color, label))

        self._virtual = True

    def toSimulation(self) -> list[int]:
        if not self._virtual:
            raise ValueError("Allowed only for virtual mixtures.")
        for p, mat, color, label in self._packs:
            self._ids[label] = set(p.toSimulation(material=mat, color=color))
        self._virtual = False
        return self.all_ids

    def toFile(self, filename: str) -> None:
        if self._virtual:
            raise ValueError("Allowed only for non-virtual mixtures.")
        open(filename, "w").close()
        with open(filename, "w") as file:
            for label, ids in self._ids.items():
                for id in ids:
                    b = O.bodies[id]
                    if not b is None:
                        pos = b.state.pos
                        r = b.shape.radius
                        file.write(f"{pos[0]} {pos[1]} {pos[2]} {r} {label}\n")

    def fromFile(
        self,
        filename: str,
        material_labels: dict[str:str],
        colors: dict[str : tuple[float, float, float]],
    ) -> None:
        spheres = dict()
        with open(filename, "r") as file:
            for line in file:
                x, y, z, r, label = line.split()
                x, y, z, r = map(float, [x, y, z, r])
                if label not in spheres:
                    spheres[label] = []
                spheres[label].append(((x, y, z), r))

        for label, sph in spheres.items():
            sp = pack.SpherePack()
            sp.fromList(sph)
            self._packs.append((sp, material_labels[label], colors[label], label))
        self._virtual = True

    def move(self, vector: Vector3) -> None:
        if not self._virtual:
            raise ValueError("Allowed only for virtual mixtures.")
        for i, elem in enumerate(self._packs):
            p, mat, color, label = elem
            l = p.toList()
            for j, s in enumerate(l):
                center, radius = s
                l[j] = (center + vector, radius)
            p.fromList(l)
            self._packs[i] = (p, mat, color, label)

    def filter(self, pred: pack.Predicate) -> None:
        """TODO"""
        if not self._virtual:
            raise ValueError("Allowed only for virtual mixtures.")
        for i, elem in enumerate(self._packs):
            p, mat, color, label = elem
            pack_pred = pack.inAlignedBox(*p.aabb())
            predicate = pack.PredicateDifference(pack_pred, pred)
            self._packs[i] = (
                pack.filterSpherePack(predicate, p, returnSpherePack=True),
                mat,
                color,
                label,
            )

    def erase(self, ids: Iterable[int]) -> None:
        """TODO"""
        if self._virtual:
            raise ValueError("Allowed only for non-virtual mixtures.")
        for label in self._ids:
            self._ids[label] = [id for id in self._ids[label] if not id in ids]
