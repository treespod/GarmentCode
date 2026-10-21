import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_TRIANGULATION_PATH = (
    Path(__file__).parent / "pygarment" / "meshgen" / "triangulation_utils.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "triangulation_utils", _TRIANGULATION_PATH
)
_TRI_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRI_UTILS)


def sample_boundary(corners, spacing):
    points = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        count = math.ceil(np.linalg.norm(end - start) / spacing)
        points.extend(start + (end - start) * step / count for step in range(count))
    return np.asarray(points)


class TrianglePanelMeshingTests(unittest.TestCase):
    def assert_panel_mesh(self, corners, expected_area, in_domain=None):
        boundary_points = sample_boundary(np.asarray(corners, dtype=float), 0.5)
        edges = [
            SimpleNamespace(vertex_range=[index, (index + 1) % len(boundary_points)])
            for index in range(len(boundary_points))
        ]
        segments = _TRI_UTILS.get_edge_vert_ids(edges)
        max_edge_length = 1.43 * 0.5
        vertices, faces = _TRI_UTILS.triangulate_panel(
            boundary_points,
            segments,
            max_edge_length=max_edge_length,
            check=True,
        )

        triangle_points = vertices[faces]
        edge_1 = triangle_points[:, 1] - triangle_points[:, 0]
        edge_2 = triangle_points[:, 2] - triangle_points[:, 0]
        face_areas = np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]) / 2
        self.assertAlmostEqual(face_areas.sum(), expected_area)
        self.assertTrue(np.all(face_areas <= np.sqrt(3) / 4 * max_edge_length ** 2 + 1e-9))
        self.assertTrue(_TRI_UTILS.is_manifold(faces, vertices))

        mesh_edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
            axis=1,
        )
        boundary_edges = np.sort(segments, axis=1)
        for boundary_edge in boundary_edges:
            self.assertTrue(np.any(np.all(mesh_edges == boundary_edge, axis=1)))

        if in_domain is not None:
            centroids = triangle_points.mean(axis=1)
            self.assertTrue(all(in_domain(point) for point in centroids))

    def test_convex_panel_preserves_boundary_and_mesh_resolution(self):
        self.assert_panel_mesh(
            [[0, 0], [2, 0], [2, 2], [0, 2]],
            expected_area=4.0,
        )

    def test_concave_panel_excludes_concavity(self):
        self.assert_panel_mesh(
            [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]],
            expected_area=5.0,
            in_domain=lambda point: point[0] <= 1 + 1e-9 or point[1] <= 1 + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
