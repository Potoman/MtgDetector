"""
Trainings read on the expansion symbol (on the right of the type line), with the steps of
training.py (see TrainingDefinition and TrainingYoloDetect there):
- set (YOLO detector): "no symbol" or the expansion code, runs/detect/set/set/<hash>/,
- set (basic CNN): the expansion code, runs/classify/set/set/<hash>/,
- rarity (basic CNN): read on the same symbol, runs/classify/rarity/rarity/<hash>/.

Run from anywhere: python3 src/training_set.py (menu of these trainings)
"""
from typing import List, Union

import training
from training import Rect, TrainingDefinition, TrainingYoloDetect


def definitions() -> List[Union[TrainingYoloDetect, TrainingDefinition]]:
    """The trainings of the expansion symbol, in the order of the menu."""
    # Filter possibilities :
    # - "frame": ["2003", "2015"]
    # Same cards for every 2015-frame training: normal layout, planeswalkers excluded.
    accept_2015 = {"frame": ["2015"], "layout": ["normal"]}
    filters_2015 = {"type_line": ["Legendary Planeswalker", "Planeswalker"]}
    return [
        # Expansion with a YOLO detector (runs/detect/set/set/): "no symbol" or the expansion code.
        # downscale: the symbol as seen by the webcam (about 35-60% of the scryfall resolution).
        TrainingYoloDetect(
            name="set",
            label="set",
            sub_categories=["rarity", "colors"],
            rect=Rect(center_x=430, center_y=402, height=64, width=64),
            accept=accept_2015,
            filters=filters_2015,
            samples_per_key=1,
            epochs=25,
            batch_size=32,
            blur_kernels=[1, 3],
            downscale=[0.35, 0.45, 0.6],
            cache="ram",  # About 0.8 GB: no slow /mnt/c read after the first epoch
        ),
        # Expansion, read on the same symbol, with the basic CNN (runs/classify/set/set/).
        # The symbol color depends on the rarity.
        TrainingDefinition(
            name="set",
            label="set",
            sub_categories=["rarity", "colors"],
            rect=Rect(center_x=430, center_y=402, height=64, width=64),
            accept=accept_2015,
            filters=filters_2015,
            samples_per_key=1,
            epochs=20,
            batch_size=32,
            blur_kernels=[1, 3],
            shift_pixels=[2, 4, 6, 8, 10, 12, 14 ,16],
        ),
        # Rarity, read on the expansion symbol, with the basic CNN. The symbol depends on the set.
        TrainingDefinition(
            name="rarity",
            label="rarity",
            sub_categories=["set"],
            rect=Rect(center_x=430, center_y=400, height=80, width=80),
            accept={"frame": ["2003", "2015"]},
        ),
    ]


if __name__ == "__main__":
    for chosen in training._choose_trainings(definitions()):
        chosen.run()
