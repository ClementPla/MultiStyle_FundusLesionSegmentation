from enum import Flag, auto


def get_dataset_from_name(name: str):
    "Return the Flag Dataset from the string name"
    for d in Dataset:
        if name == d.name:
            return d


class Dataset(Flag):
    IDRID = auto()
    RETINAL_LESIONS = auto()
    MESSIDOR = auto()
    FGADR = auto()
    KAGGLE_TEACHER = auto()
    DDR = auto()

    @property
    def name(cls):
        name = super(Dataset, cls).name
        if name:
            return name
        else:
            return [flag.name for flag in Dataset if flag in cls]

    @property
    def str_name(cls):
        name = cls.name
        if isinstance(name, list):
            return "_".join(name)
        else:
            return name

    @property
    def suffix(cls):
        name = cls.name
        if isinstance(name, list):
            return ["_%s" % f.lower() for f in cls.name]
        else:
            return "_%s" % name.lower()

    @property
    def length(cls):
        name = cls.name
        if isinstance(name, list):
            return len(name)
        else:
            return 1

    def __len__(self):
        return self.length


class Lesions(Flag):
    COTTON_WOOL_SPOT = auto()
    EXUDATES = auto()
    HEMORRHAGES = auto()
    MICROANEURYSMS = auto()

    @property
    def name(cls):
        name = super(Lesions, cls).name
        if name:
            return name
        else:
            return [flag.name for flag in Lesions if flag in cls]

    @property
    def str_name(cls):
        name = cls.name
        if isinstance(name, list):
            return "_".join(name)
        else:
            return name

    @property
    def length(cls):
        name = cls.name
        if isinstance(name, list):
            return len(name)
        else:
            return 1

    def __len__(self):
        return self.length


ALL_DATASETS = Dataset.IDRID | Dataset.MESSIDOR | Dataset.DDR | Dataset.FGADR | Dataset.RETINAL_LESIONS
ALL_CLASSES = Lesions.COTTON_WOOL_SPOT | Lesions.EXUDATES | Lesions.HEMORRHAGES | Lesions.MICROANEURYSMS
