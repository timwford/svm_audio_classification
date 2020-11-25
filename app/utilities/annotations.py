from typing import Annotated

from numpy import ndarray

Seconds = Annotated[int, 'seconds']
Recording = Annotated[ndarray, 'recording']
