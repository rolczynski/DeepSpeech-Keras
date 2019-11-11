import itertools
from typing import Tuple, List
from collections import defaultdict
import numpy as np


def edit_distance(source: List[str], destination: List[str]) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Calculation of edit distance between two sequences.

    This is the Levenshtein distance with the substitution cost equals 1.
    It is the iterative method with the full matrix support.
    O(nm) time and space complexity.

    References:
        - https://web.stanford.edu/class/cs124/lec/med.pdf
        - https://www.python-course.eu/levenshtein_distance.php
        - https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
        - https://giovanni.carmantini.com/2016/01/minimum-edit-distance-in-python.html
    """
    size_x = len(source) + 1
    size_y = len(destination) + 1
    matrix = np.zeros([size_x, size_y])
    matrix[:, 0] = np.arange(0, size_x)
    matrix[0, :] = np.arange(0, size_y)
    backtrace = np.zeros_like(matrix, dtype=[('del', bool),
                                             ('sub', bool),
                                             ('ins', bool),
                                             ('cost', int)])
    backtrace[:, 0] = (True, False, False, 0)
    backtrace[0, :] = (False, False, True, 0)
    for x, y in itertools.product(range(1, size_x),
                                  range(1, size_y)):
        if source[x-1] == destination[y-1]:
            cost = 0
        else:
            cost = 1
        delete = matrix[x-1][y] + 1
        insert = matrix[x][y-1] + 1
        substitute = matrix[x-1][y-1] + cost
        min_dist = min(delete, insert, substitute)
        matrix[x, y] = min_dist
        backtrace[x, y] = (delete == min_dist,
                           substitute == min_dist,
                           insert == min_dist,
                           cost)
    return matrix[size_x-1, size_y-1], matrix, backtrace


def simple_backtrace(backtrace: np.ndarray):
    """ Calculate the editing path via the backtrace. """
    rows, columns = backtrace.shape
    i, j = rows-1, columns-1
    backtrace_indices = [(i, j, 'sub', 0)]
    while (i, j) != (0, 0):
        delete, substitute, insert, cost = backtrace[i, j]
        if insert:
            operation = 'ins'
            i, j = i, j-1
        elif substitute:
            operation = 'sub'
            i, j = i-1, j-1
        elif delete:
            operation = 'del'
            i, j = i - 1, j
        else:
            raise KeyError("Backtrace matrix wrong defined")
        backtrace_indices.append((i, j, operation, cost))
    return list(reversed(backtrace_indices))


def decode_path(best_path: List[Tuple[int, int, str, int]], source: List[str], destination: List[str]):
    """ Collect all transformations needed to go from `source` to `destination`. """
    to_delete, to_insert, to_substitute = [], [], defaultdict(list)
    for index, (i, j, operation, cost) in enumerate(best_path):
        if operation == 'del':
            item = source[i]
            to_delete.append(item)
        elif operation == 'sub' and cost:   # without cost sub operation indicates correctness
            wrong_item, target_item = source[i], destination[j]
            to_substitute[target_item].append(wrong_item)
        elif operation == 'ins':
            item = destination[j]
            to_insert.append(item)
    return to_delete, to_insert, to_substitute
