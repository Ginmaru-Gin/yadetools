import numpy as np
from yade import Vector3

from yadetools.mixture import Mixture
from yadetools.tools import sphere_Vr


def test_mixture_parts():
    def get_parts(mixture: Mixture, dens: list[float] | None =  None, by_mass: bool = False) -> list[float]:
        values = []
        for p, _, _, _ in mixture._packs:
            values.append(np.sum([sphere_Vr(i[1]) for i in p]))
        if by_mass:
            if dens is None:
                raise ValueError("'dens' is necessary argument with argument 'by_mass'")
            values = [v * d for v, d in zip(values, dens)]
        s = np.sum(values)
        return [v / s for v in values]

    min_corner = Vector3(0, 0, 0)
    max_corner = Vector3(1, 1, 1)
    labels = ["first", "second"]
    parts = [[0.1, 0.3, 0.5, 0.8, 1], [0.2, 0.5, 1]]
    psd_sizes = [[0.02, 0.022, 0.024, 0.026, 0.029], [0.01, 0.012, 0.015]]
    mix_density = [2.65e3, 1.2e3]
    materials = ["mat1", "mat2"]
    disp = 0.5

    mix = Mixture()
    mix.generate(
        min_corner,
        max_corner,
        grow_axis="z",
        num=100_000,
        labels=labels,
        # parts=parts,
        # psd_sizes=psd_sizes,
        parts=[[0.1, 0.5, 1], None],
        psd_sizes=[[6.35 * 0.001, 7.8 * 0.001, 9.53 * 0.001], 2.8e-3],
        mix_parts=[0.7, 0.3],
        mix_density=mix_density,
        material_labels=materials,
        disp=disp,
        distribute_mass=True,
    )
    mix_parts = get_parts(mix)

    assert 0.68 < mix_parts[0] < 0.72
    assert 0.28 < mix_parts[1] < 0.32

    mix = Mixture()
    mix.generate(
        min_corner,
        max_corner,
        grow_axis="z",
        num=100_000,
        labels=labels,
        # parts=parts,
        # psd_sizes=psd_sizes,
        parts=[[0.1, 0.5, 1], None],
        psd_sizes=[[6.35 * 0.001, 7.8 * 0.001, 9.53 * 0.001], 2.8e-3],
        mix_parts=[0.7, 0.3],
        mix_density=mix_density,
        material_labels=materials,
        disp=disp,
        distribute_mass=True,
        mix_by_mass=True,
    )
    mix_parts = get_parts(mix, mix_density, True)

    assert 0.68 < mix_parts[0] < 0.72
    assert 0.28 < mix_parts[1] < 0.32
