"""Weighted tree edit distance between two ARIEL robot body graphs.

This is the fitness function for the morphological-adaptation assignment.
You do not have to modify it, but you SHOULD read it - you are about to spend
a lot of compute optimising the number it returns, and you cannot reason about
what your EA is doing if you do not know what it is being scored on.

WHAT IT MEASURES
----------------
The cheapest sequence of edits that turns one body into the other, where an
edit is one of:

    delete a module        cost 1.0
    insert a module        cost 1.0
    relabel a module       cost 0.0  if type AND rotation match
                           cost 0.5  if type matches but rotation does not
                           cost 1.0  if the type does not match

Distance 0.0 means the two bodies are structurally identical.

HOW IT WORKS
------------
An ARIEL body graph is a rooted tree: one core, and every other module hangs
off exactly one parent. So this uses the Zhang-Shasha tree edit distance
algorithm, which is O(n^2 * d^2) - milliseconds per comparison, as opposed to
general graph edit distance, which is exponential.

Children are ordered canonically by attachment face (FRONT, BACK, RIGHT, LEFT,
TOP, BOTTOM) so that the same body always produces the same tree.
"""

# Graph node ids are whatever the decoder chose, so they really are Any
# (ANN401); Zhang-Shasha genuinely needs its table variables (PLR0914).
# ruff: noqa: ANN401, DOC201, DOC402, PLR0914

# Standard library
from collections.abc import Callable, Iterator
from typing import Any

# Third-party libraries
import networkx as nx

# --- COST CONSTANTS --- #
# Change these and the shape of the search landscape changes with them.
# If you do, say so in your report and show what it did - don't change those unless that's what you're researching.
DELETE_COST: float = 1.0
INSERT_COST: float = 1.0
TYPE_MISMATCH_COST: float = 1.0
ROTATION_MISMATCH_COST: float = 0.5

# Canonical child ordering. Any fixed order works; this one is the enum order.
FACE_ORDER: tuple[str, ...] = (
    "FRONT",
    "BACK",
    "RIGHT",
    "LEFT",
    "TOP",
    "BOTTOM",
)
_FACE_RANK: dict[str, int] = {face: i for i, face in enumerate(FACE_ORDER)}

# A node label is the pair we compare on.
type NodeLabel = tuple[str, str]  # (module type, rotation)


# ============================================================================ #
#  Turning a body graph into an ordered tree
# ============================================================================ #


def _find_root(graph: nx.DiGraph) -> Any:
    """Return the unique node with no parent (the core).

    Raises
    ------
    ValueError
        If the graph is empty or does not have exactly one root.
    """
    roots = [n for n, deg in graph.in_degree() if deg == 0]
    if len(roots) != 1:
        msg = f"expected exactly one root module, found {len(roots)}"
        raise ValueError(msg)
    return roots[0]


def _ordered_children(graph: nx.DiGraph, node: Any) -> list[Any]:
    """Return a node's children sorted by attachment face, then by id."""
    return sorted(
        graph.successors(node),
        key=lambda child: (
            _FACE_RANK.get(
                graph.edges[node, child].get("face", ""),
                len(FACE_ORDER),
            ),
            str(child),
        ),
    )


def _label_of(graph: nx.DiGraph, node: Any) -> NodeLabel:
    """Return the (type, rotation) label used for comparison."""
    attrs = graph.nodes[node]
    return (str(attrs.get("type", "")), str(attrs.get("rotation", "")))


def _postorder(graph: nx.DiGraph) -> tuple[list[NodeLabel], list[int]]:
    """Flatten a body graph into post-order labels and leftmost-descendants.

    Returns
    -------
    labels : list of (type, rotation)
        Node labels in post-order.
    leftmost : list of int
        `leftmost[i]` is the post-order index of the leftmost leaf descendant
        of node `i` - the bookkeeping Zhang-Shasha needs.
    """
    labels: list[NodeLabel] = []
    leftmost: list[int] = []

    def visit(node: Any) -> int:
        first_child_leftmost: int | None = None
        for child in _ordered_children(graph, node):
            child_leftmost = visit(child)
            if first_child_leftmost is None:
                first_child_leftmost = child_leftmost

        labels.append(_label_of(graph, node))
        my_index = len(labels) - 1
        my_leftmost = (
            first_child_leftmost
            if first_child_leftmost is not None
            else my_index
        )
        leftmost.append(my_leftmost)
        return my_leftmost

    visit(_find_root(graph))
    return labels, leftmost


def _keyroots(leftmost: list[int]) -> list[int]:
    """Return the key roots: the largest node index per leftmost descendant."""
    seen: set[int] = set()
    roots: list[int] = []
    for i in range(len(leftmost) - 1, -1, -1):
        if leftmost[i] not in seen:
            seen.add(leftmost[i])
            roots.append(i)
    roots.reverse()
    return roots


# ============================================================================ #
#  The cost of turning one module into another
# ============================================================================ #


def relabel_cost(label_a: NodeLabel, label_b: NodeLabel) -> float:
    """Cost of turning module `label_a` into module `label_b`."""
    type_a, rotation_a = label_a
    type_b, rotation_b = label_b

    if type_a != type_b:
        return TYPE_MISMATCH_COST
    if rotation_a != rotation_b:
        return ROTATION_MISMATCH_COST
    return 0.0


# ============================================================================ #
#  Zhang-Shasha
# ============================================================================ #


def tree_edit_distance(
    graph_a: nx.DiGraph,
    graph_b: nx.DiGraph,
    cost_fn: Callable[[NodeLabel, NodeLabel], float] = relabel_cost,
) -> float:
    """Return the weighted tree edit distance between two body graphs.

    Parameters
    ----------
    graph_a, graph_b : nx.DiGraph
        Robot body graphs. Nodes carry `type` and `rotation` attributes,
        edges carry a `face` attribute - exactly what every ARIEL decoder
        produces.
    cost_fn : callable, optional
        Relabel cost between two (type, rotation) labels. Defaults to
        `relabel_cost`.

    Returns
    -------
    float
        0.0 when the bodies are structurally identical; lower is better.
    """
    labels_a, leftmost_a = _postorder(graph_a)
    labels_b, leftmost_b = _postorder(graph_b)
    size_a, size_b = len(labels_a), len(labels_b)

    # treedist[i][j] = edit distance between subtree at i and subtree at j
    treedist = [[0.0] * size_b for _ in range(size_a)]

    for i in _keyroots(leftmost_a):
        for j in _keyroots(leftmost_b):
            rows = i - leftmost_a[i] + 2
            cols = j - leftmost_b[j] + 2
            row_offset = leftmost_a[i] - 1
            col_offset = leftmost_b[j] - 1

            # forest distance table for this pair of key roots
            forest = [[0.0] * cols for _ in range(rows)]
            for x in range(1, rows):
                forest[x][0] = forest[x - 1][0] + DELETE_COST
            for y in range(1, cols):
                forest[0][y] = forest[0][y - 1] + INSERT_COST

            for x in range(1, rows):
                for y in range(1, cols):
                    node_a = x + row_offset
                    node_b = y + col_offset

                    delete = forest[x - 1][y] + DELETE_COST
                    insert = forest[x][y - 1] + INSERT_COST

                    both_are_whole_subtrees = (
                        leftmost_a[node_a] == leftmost_a[i]
                        and leftmost_b[node_b] == leftmost_b[j]
                    )
                    if both_are_whole_subtrees:
                        change = forest[x - 1][y - 1] + cost_fn(
                            labels_a[node_a],
                            labels_b[node_b],
                        )
                        forest[x][y] = min(delete, insert, change)
                        treedist[node_a][node_b] = forest[x][y]
                    else:
                        p = leftmost_a[node_a] - 1 - row_offset
                        q = leftmost_b[node_b] - 1 - col_offset
                        change = forest[p][q] + treedist[node_a][node_b]
                        forest[x][y] = min(delete, insert, change)

    return treedist[size_a - 1][size_b - 1]




def mean_plus_std_tree_edit_distance(
    graph: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Return mean edit distance plus the (population) std across targets.

    This is the assignment's fitness: a single body is scored against the
    whole target set at once, so the EA has to find a morphology that is a
    good compromise across all of them.
    The std term penalises uneven per-target distances directly, so nailing
    one target while ignoring the rest costs more than the mean alone would
    charge.

    Raises
    ------
    ValueError
        If `targets` is empty.
    """
    if not targets:
        msg = "cannot score against an empty target set"
        raise ValueError(msg)
    dists = [tree_edit_distance(graph, target) for target in targets]
    mean = sum(dists) / len(dists)
    variance = sum((d - mean) ** 2 for d in dists) / len(dists)
    return mean + variance**0.5
    

def distances_to_targets(
    graph: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> Iterator[float]:
    """Yield the distance to each target in turn.

    Useful for reporting: the mean alone hides whether your body is mediocre
    against every target or excellent against one and hopeless against the
    rest.
    """
    for target in targets:
        yield tree_edit_distance(graph, target)
