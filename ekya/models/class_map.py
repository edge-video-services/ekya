# This file contains a bunch of maps translating classes between different datasets, so you can use imagenet model on cityscapes etc.


# Cityscapes map:
# classes_of_interest_map = {'person': 0,
#                                  'car': 1,
#                                  'truck': 2,
#                                  'bus': 3,
#                                  'bicycle': 4,
#                                  'motorcycle': 5
#                                            }
CITYSCAPES_TO_IMAGENET_LISTING = {
    0: [],
    1: [274, 276, 287, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 286],
    2: [265, 279, 280, 281, 282, 283, 284, 285, 289, 290],
    3: [256, 257],
    4: [254, 255, 291, 292, ],
    5: [260, 277]
}