import os

from fundseg.data.utils import Dataset

_all_datasets = [Dataset.IDRID, Dataset.MESSIDOR, Dataset.DDR, Dataset.FGADR, Dataset.RETINAL_LESIONS]

def get_class_mapping(datasets=_all_datasets):
    return {k.value: i for i, k in enumerate(datasets)}

def get_reverse_class_mapping(datasets=_all_datasets):
    return {i: k for i, k in enumerate(datasets)}

runs_ids = {
    5: "deep-brook-44",
    4: "driven-thunder-43",
    3: "silver-wave-42",
    2: "dutiful-vortex-41",
    1: "ethereal-puddle-40",
    0: "earnest-lion-39",
    -1: "efficient-butterfly-50",
    Dataset.IDRID | Dataset.RETINAL_LESIONS: "smart-silence-53",
}


def best_modelpath_by_id(model_id):
    folder = runs_ids[model_id]
    model_folder = os.path.join(root, folder)
    list_model = os.listdir(model_folder)
    best_model = next(_ for _ in list_model if "epoch" in _)
    return os.path.join(model_folder, best_model)


root = "checkpoints_probing/"
trained_probe_path = {k: best_modelpath_by_id(k) for k in runs_ids.keys()}
