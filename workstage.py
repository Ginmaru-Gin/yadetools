import json
from yade import O, utils
from yade.utils import Vector3
from yadetools.mixture import Mixture


def load_spheres(filename: str, substances: list[dict], predicates: list = []) -> Mixture:
    mat_labels = {s["label"]: s["mat_label"] for s in substances}
    colors = {s["label"]: s["color"] for s in substances}
    spheres = Mixture()
    spheres.fromFile(
        filename,
        mat_labels,
        colors,
    )
    for pred in predicates:
        spheres.filter(pred)
    return spheres


def save_params(filename: str, classes: list):
    def is_jsonable(x):
        try:
            json.dumps(x)
        except:
            return False
        else:
            return True

    with open(filename, "w") as file:
        for cls in classes:
            dump = {
                k: v
                for k, v in vars(cls).items()
                if not k.startswith("__")
                and is_jsonable(v)
            }
            dump = json.dumps(dump)
            file.write("=>".join([cls.__name__, dump]) + "\n")


def load_params(filename:str, classes: list, exclude_attrs: dict):
    with open(filename, "r") as file:
        for line in file:
            cls_name, dump = line.split("=>")
            dump = json.loads(dump)
            cls = next(cls for cls in classes if cls.__name__ == cls_name)
            for k, v in dump.items():
                if k not in exclude_attrs:
                    setattr(cls, k, v)
    