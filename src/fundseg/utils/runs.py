import os

from fundseg.data.utils import ALL_DATASETS, Dataset

runs_id = {
    ALL_DATASETS: "confused-water-37",
    Dataset.IDRID: "evil-goblin-39",
    Dataset.MESSIDOR: "mischievous-spell-40",
    Dataset.DDR: "dark-phantasm-41",
    Dataset.FGADR: "haunted-clown-42",
    Dataset.RETINAL_LESIONS: "phantom-poltergeist-43",
    Dataset.IDRID | Dataset.RETINAL_LESIONS: "crisp-cosmos-72",
}

root = "checkpoints/"


def best_modelpath_by_id(dataset_id):
    folder = runs_id[dataset_id]
    model_folder = os.path.join(root, folder)
    list_model = os.listdir(model_folder)
    best_model = next(_ for _ in list_model if "epoch" in _)
    return os.path.join(model_folder, best_model)


models_path = {k: best_modelpath_by_id(k) for k in runs_id.keys()}
