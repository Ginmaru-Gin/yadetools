import os
import sys
from subprocess import Popen
from time import sleep
from yade import O, utils
from yade.utils import Vector3

from .mixture import Mixture


def define_materials(*materials):
    for mat in materials:
        O.materials.append(mat)


def create_box(center: Vector3, size: Vector3, material, mask: int):
    return utils.aabbWalls(
        (center - 0.5 * size, center + 0.5 * size),
        material=material,
        mask=mask,
    )


def create_spheres(
    center: Vector3,
    size: Vector3,
    num: int,
    substances: list[dict],
    *,
    grow_axis: str = "z",
    disp: float = 0,
    mix_by_mass: bool = True,
    seed: int = -1,
) -> Mixture:
    mixture = Mixture()
    mixture.generate(
        center - 0.5 * size,
        center + 0.5 * size,
        num=num,
        grow_axis=grow_axis,
        labels=[s["label"] for s in substances],
        mix_parts=[s["mix_part"] for s in substances],
        parts=[s["part"] for s in substances],
        psd_sizes=[s["psd_size"] for s in substances],
        mix_density=[s["density"] for s in substances],
        material_labels=[s["mat_label"] for s in substances],
        colors=[s["color"] for s in substances],
        seed=seed,
        disp=disp,
        mix_by_mass=mix_by_mass,
    )
    return mixture


def run_parallel_layers(number: int, worker_number: int) -> list[str]:
    filenames = [f"spheres_{i}" for i in range(number)]
    log_filenames = {f + ".log" for f in filenames}
    for filename in filenames:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
    procs = [
        Popen(
            [
                "yade",
                "-nxj " + str(worker_number // number),
                sys.argv[0],
                str(i),
                ">",
                log_file,
            ]
        )
        for i, log_file in zip(range(number), log_filenames)
    ]
    for proc in procs:
        while proc.poll() is None:
            sleep(1)
        proc.terminate()

    return filenames


def load_layers(filenames: list[str], substances: list[dict]) -> list[Mixture]:
    layers = []
    mat_labels = {s["label"]: s["mat_label"] for s in substances}
    colors = {s["label"]: s["color"] for s in substances}
    for filename in filenames:
        layer = Mixture()
        layer.fromFile(
            filename,
            mat_labels,
            colors,
        )
        layers.append(layer)
    return layers


def add_layers(mixture: Mixture, layers: list[Mixture], grow_axis: int) -> None:
    for layer in layers:
        curr_height = max(
            [
                O.bodies[id].state.pos[grow_axis]
                + O.bodies[id].shape.radius
                for id in mixture.all_ids
            ]
        )
        move_vector = Vector3.Zero
        move_vector[grow_axis] += curr_height
        layer.move(move_vector)
        layer.toSimulation()
        mixture += layer
    return mixture
