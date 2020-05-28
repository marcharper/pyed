import os


def bomze_matrices(filename="bomze.txt"):
    """
    Yields the 48 matrices from I.M. Bomze's classification of three player phase
    portraits.

    Bomze, Immanuel M. "Lotka-Volterra equation and replicator dynamics: new issues in classification."
    Biological cybernetics 72.5 (1995): 447-453.
    """

    this_dir, this_filename = os.path.split(__file__)

    handle = open(os.path.join(this_dir, filename))
    for line in handle:
        a, b, c, d, e, f, g, h, i = map(float, line.split())
        yield [[a, b, c], [d, e, f], [g, h, i]]
