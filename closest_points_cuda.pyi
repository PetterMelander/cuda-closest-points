import numpy
import numpy.typing
import typing

class Pair:
    ax: int
    ay: int
    bx: int
    by: int
    distance: int
    def __init__(self) -> None: ...

def closest_points(image: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> Pair: ...
