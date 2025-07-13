import numpy
import numpy.typing
import typing

class Pair:
    a: int
    ax: int
    ay: int
    b: int
    bx: int
    by: int
    distance: float
    def __init__(self) -> None: ...

def closest_points(image: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> list[list[Pair]]: ...
