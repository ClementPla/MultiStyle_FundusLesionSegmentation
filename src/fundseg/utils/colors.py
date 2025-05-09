import numpy as np
from matplotlib.colors import ListedColormap
from PIL import ImageColor

from fundseg.data.utils import ALL_CLASSES

TRANSPARENT = "#000000"
TEAL0 = "#9ec5d4"
TEAL1 = "#75a8ba"
TEAL2 = "#4b8a9f"
TEAL3 = "#277088"
TEAL4 = "#14597e"

PURPLE00 = "#beadd1"
PURPLE0 = "#b8a1d1"
PURPLE1 = "#a17bb8"
PURPLE1bis = "#927dc0"
PURPLE2 = "#8e5ea1"
PURPLE2bis = "#7f5fa9"
PURPLE3 = "#7b3e8a"
PURPLE4 = "#6a1e75"


COLORS = [TRANSPARENT, TEAL0, TEAL4, PURPLE2, PURPLE4]
COLORS_RGB = [ImageColor.getrgb(color) for color in COLORS]
CLASSES_COLORS = {label: color for label, color in zip(["BG", *[_.name for _ in ALL_CLASSES]], COLORS)}
CMAP = ListedColormap(np.asarray(COLORS_RGB) / 255.0, N=len(COLORS_RGB))
