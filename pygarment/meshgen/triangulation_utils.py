"""Helper functions for triangulating garment panels."""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr


def get_edge_vert_ids(edges):
    """Return endpoint index pairs for every panel boundary segment."""
    segments = np.empty((0, 2), dtype=int)
    for edge in edges:
        vertex_ids = edge.vertex_range
        previous_ids = np.roll(vertex_ids, 1, axis=0)
        segments = np.concatenate(
            (segments, np.stack((previous_ids, vertex_ids), axis=1)[1:]),
            axis=0,
        )
    return segments


def triangulate_panel(points, segments, max_edge_length, plot=False, check=False):
    """Triangulate a panel PSLG while preserving its input boundary segments."""
    points = np.asarray(points, dtype=float)
    segments = np.asarray(segments, dtype=int)
    if len(points) < 3 or len(segments) < 3:
        raise ValueError("A panel needs at least three points and boundary segments")
    if max_edge_length <= 0:
        raise ValueError("max_edge_length must be positive")

    max_area = np.sqrt(3) / 4 * max_edge_length ** 2
    result = tr.triangulate(
        {"vertices": points, "segments": segments},
        f"pq14.5a{max_area:.12g}YQ",
    )
    vertices = result.get("vertices", np.empty((0, 2)))
    faces = result.get("triangles", np.empty((0, 3), dtype=int))

    if not len(faces):
        raise ValueError("Triangle did not produce any panel faces")
    if check and not np.array_equal(vertices[:len(points)], points):
        raise ValueError("Triangle changed the input panel vertex coordinates")

    if plot:
        _plot_triangulation(vertices, faces, segments)

    return vertices, faces


def _plot_triangulation(vertices, faces, segments):
    plt.triplot(vertices[:, 0], vertices[:, 1], faces, color="blue")
    for start, end in segments:
        boundary = vertices[[start, end]]
        plt.plot(boundary[:, 0], boundary[:, 1], "r-")
    plt.axis("equal")
    plt.show()


def is_manifold(face_v_ids: np.ndarray, points: np.ndarray, tol=1e-2):
    """Check that every mesh face satisfies the triangle inequality."""
    faces = points[face_v_ids]
    face_side_1 = np.linalg.norm(faces[:, 0] - faces[:, 1], axis=1)
    face_side_2 = np.linalg.norm(faces[:, 1] - faces[:, 2], axis=1)
    face_side_3 = np.linalg.norm(faces[:, 0] - faces[:, 2], axis=1)
    side_lengths = np.stack([face_side_1, face_side_2, face_side_3], axis=-1)
    return np.all(side_lengths.sum(axis=1) > 2 * side_lengths.max(axis=1) + tol)
