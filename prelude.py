from typing import Iterator, Tuple
from itertools import pairwise
import numpy as np

def grid_pairs(grid: np.ndarray) -> Iterator[Tuple[int, int]]:
    l = int(np.sqrt(len(grid)))
    assert l ** 2 == len(grid), "Grid must be square"
    for row in range(l):
        for i, j in pairwise(grid[row*l:(row+1)*l]):
            yield i, j

    for k in range(l * (l - 1)):
        i = grid[k]
        j = grid[k + l]
        yield i, j

if __name__ == "__main__":
    grid = np.array(range(9))
    pairs = grid_pairs(grid)
    for pair in pairs:
        print(pair)
