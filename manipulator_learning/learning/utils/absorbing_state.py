""" Utilties for absorbing states from DAC (Discriminator Actor Critic) paper """
from enum import Enum


class Mask(Enum):
    ABSORBING = -1.0
    DONE = 0.0
    NOT_DONE = 1.0
