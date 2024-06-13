from enum import Flag, auto


class Lesions(str, Flag):
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


ALL_CLASSES = [
    Lesions.COTTON_WOOL_SPOT.name,
    Lesions.EXUDATES.name,
    Lesions.HEMORRHAGES.name,
    Lesions.MICROANEURYSMS.name,
]
