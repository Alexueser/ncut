import numpy as np
import matplotlib.pyplot as plt
from gen_data_points import make_points
from solve_ncuts import n_cuts


def compute_connection(points):
    n_items = len(points)

    distance = np.zeros((n_items, n_items), dtype=np.float)
    for which in range(n_items):
        distance[which, :] = np.sqrt(
            np.power(points[:, 0] - points[which, 0], 2) +
            np.power(points[:, 1] - points[which, 1], 2)
        )

    scale_sig = np.max(distance)
    W = np.exp(-np.power(distance / scale_sig, 2))
    return W


def segment_and_show(points, eigenvalues, eigenvectors):
    abs_eigenvalues = np.abs(eigenvalues)
    print(abs_eigenvalues[abs_eigenvalues.argsort()])
    # Only show 4 graphs
    for pos in abs_eigenvalues.argsort()[1:min(5, len(abs_eigenvalues))]:
        eigenvector = eigenvectors[pos] > 0
        points_a = points[eigenvector]
        points_b = points[~eigenvector]

        # Adjust image form
        plt.figure()
        plt.grid(True)
        plt.axis('equal')

        # Show data
        plt.plot(points_a[:, 0], points_a[:, 1], 'ro')
        plt.plot(points_b[:, 0], points_b[:, 1], 'bo')

        plt.show(block=False)
        print("Collection A: {}".format(len(points_a)))
        print("Collection B: {}".format(len(points_b)))
        print("Item Select: {}".format(pos))
        print("Value: {}".format(abs_eigenvalues[pos]))
        print("=" * 10)


if __name__ == '__main__':
    # 1: Fixed 16 points
    # 2: Two parts
    # 3: Three parts
    points = make_points(3)
    W = compute_connection(points)
    (eigenvalues, eigenvectors) = n_cuts(W, regularization=False)
    segment_and_show(points, eigenvalues, eigenvectors)
    input("Press ENTER to continue...")