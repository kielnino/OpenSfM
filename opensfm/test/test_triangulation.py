# pyre-strict
import numpy as np
from numpy.typing import ArrayLike, NDArray
from opensfm import io, pygeometry, pymap, reconstruction


def test_track_triangulator_spherical() -> None:
    """Test triangulating tracks of spherical images."""
    tracks_manager = pymap.TracksManager()
    tracks_manager.add_observation("im1", "1", pymap.Observation(0, 0, 1.0, 0, 0, 0, 0))
    tracks_manager.add_observation(
        "im2", "1", pymap.Observation(-0.1, 0, 1.0, 0, 0, 0, 1)
    )

    rec = io.reconstruction_from_json(
        {
            "cameras": {
                "theta": {
                    "projection_type": "spherical",
                    "width": 800,
                    "height": 400,
                }
            },
            "shots": {
                "im1": {
                    "camera": "theta",
                    "rotation": [0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                },
                "im2": {
                    "camera": "theta",
                    "rotation": [0.0, 0.0, 0.0],
                    "translation": [-1.0, 0.0, 0.0],
                },
            },
            "points": {},
        }
    )

    triangulator = reconstruction.TrackTriangulator(
        rec, reconstruction.TrackHandlerTrackManager(tracks_manager, rec)
    )
    triangulator.triangulate(
        "1",
        reproj_threshold=0.01,
        min_ray_angle_degrees=2.0,
        min_depth=0.001,
        iterations=10,
    )
    assert "1" in rec.points
    p = rec.points["1"].coordinates
    assert np.allclose(p, [0, 0, 1.3763819204711])
    assert len(rec.points["1"].get_observations()) == 2


def test_track_triangulator_coincident_camera_origins() -> None:
    """Test triangulating tracks when two cameras have the same origin.

    Triangulation should fail and no points should be added to the reconstruction.
    """
    tracks_manager = pymap.TracksManager()
    tracks_manager.add_observation("im1", "1", pymap.Observation(0, 0, 1.0, 0, 0, 0, 0))
    tracks_manager.add_observation(
        "im2", "1", pymap.Observation(-0.1, 0, 1.0, 0, 0, 0, 1)
    )

    rec = io.reconstruction_from_json(
        {
            "cameras": {
                "theta": {
                    "projection_type": "spherical",
                    "width": 800,
                    "height": 400,
                }
            },
            "shots": {
                "im1": {
                    "camera": "theta",
                    "rotation": [0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                },
                "im2": {
                    "camera": "theta",
                    "rotation": [0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                },
            },
            "points": {},
        }
    )

    triangulator = reconstruction.TrackTriangulator(
        rec, reconstruction.TrackHandlerTrackManager(tracks_manager, rec)
    )
    triangulator.triangulate(
        "1",
        reproj_threshold=0.01,
        min_ray_angle_degrees=2.0,
        min_depth=0.0001,
        iterations=10,
    )
    assert not rec.points


def unit_vector(x: ArrayLike) -> NDArray:
    return np.array(x) / np.linalg.norm(x)


def test_triangulate_bearings_dlt() -> None:
    rt1 = np.append(np.identity(3), [[0], [0], [0]], axis=1)
    rt2 = np.append(np.identity(3), [[-1], [0], [0]], axis=1)
    b1 = unit_vector([0.0, 0, 1])
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    res, X = pygeometry.triangulate_bearings_dlt(
        [rt1, rt2], np.asarray([b1, b2]), max_reprojection, min_ray_angle, min_depth
    )
    assert np.allclose(X, [0, 0, 1.0])
    assert res is True


def test_triangulate_bearings_dlt_coincident_camera_origins() -> None:
    rt1 = np.append(np.identity(3), [[0], [0], [0]], axis=1)
    rt2 = np.append(np.identity(3), [[0], [0], [0]], axis=1)  # same origin
    b1 = unit_vector([0.0, 0, 1])
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    res, X = pygeometry.triangulate_bearings_dlt(
        [rt1, rt2], np.asarray([b1, b2]), max_reprojection, min_ray_angle, min_depth
    )
    assert res is False


def test_triangulate_bearings_midpoint() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([1.0, 0, 0])
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
        np.asarray([o1, o2]),
        np.asarray([b1, b2]),
        2 * [max_reprojection],
        min_ray_angle,
        min_depth,
    )
    assert np.allclose(X, [0, 0, 1.0])
    assert valid_triangulation is True


def test_triangulate_bearings_midpoint_coincident_camera_origins() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([0.0, 0, 0])  # same origin
    b2 = unit_vector([-1.0, 0, 1])
    max_reprojection = 0.01
    min_ray_angle = np.radians(2.0)
    min_depth = 0.001
    valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
        np.asarray([o1, o2]),
        np.asarray([b1, b2]),
        2 * [max_reprojection],
        min_ray_angle,
        min_depth,
    )
    assert valid_triangulation is False


def test_triangulate_two_bearings_midpoint() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([1.0, 0, 0])
    b2 = unit_vector([-1.0, 0, 1])
    ok, X = pygeometry.triangulate_two_bearings_midpoint(
        np.asarray([o1, o2]), np.asarray([b1, b2])
    )
    assert ok is True
    assert np.allclose(X, [0, 0, 1.0])


def test_triangulate_two_bearings_midpoint_failed() -> None:
    o1 = np.array([0.0, 0, 0])
    b1 = unit_vector([0.0, 0, 1])
    o2 = np.array([1.0, 0, 0])

    # almost parallel. 1e-5 will make it triangulate again.
    b2 = b1 + np.array([-1e-10, 0, 0])

    ok, X = pygeometry.triangulate_two_bearings_midpoint(
        np.asarray([o1, o2]), np.asarray([b1, b2])
    )
    assert ok is False


def test_epipolar_angle_fisheye_opencv() -> None:
    """Test that epipolar_angle_two_bearings_many works correctly for FishEyeOpencvCamera.

    Sets up two cameras with a known relative pose and projects ground-truth 3D
    points through both cameras.  The resulting bearings must satisfy the
    epipolar constraint: corresponding pairs (diagonal) should yield a
    near-zero angle, while non-corresponding pairs (off-diagonal) should yield
    a clearly positive angle.
    """
    focal = 0.5
    aspect_ratio = 1.0
    principal_point = np.array([0.0, 0.0])
    distortion = np.array([0.1, 0.01, 0.001, 0.0001])
    camera = pygeometry.Camera.create_fisheye_opencv(
        focal, aspect_ratio, principal_point, distortion
    )
    camera.width = 1000
    camera.height = 1000

    # 3D ground-truth points in world frame (all in front of both cameras).
    # Camera 1 is at the world origin looking along +z, so world frame ==
    # camera-1 frame.
    gt_points = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.5, 0.2, 4.0],
            [-0.3, 0.4, 3.5],
            [0.1, -0.5, 5.0],
        ]
    )

    # Camera 2 is translated along x by 1 unit with the same orientation.
    # Relative pose: R_cam_to_world = I,  origin_in_world = [1, 0, 0].
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])

    # Bearings in camera 1's frame (= world frame).
    pixels1 = camera.project_many(gt_points)
    b1 = camera.pixel_bearing_many(pixels1)

    # Bearings in camera 2's frame (translate world points into cam-2 frame).
    points_in_cam2 = gt_points - translation
    pixels2 = camera.project_many(points_in_cam2)
    b2 = camera.pixel_bearing_many(pixels2)

    n = len(gt_points)

    # Verify with float64 bearings.
    angles_f64 = pygeometry.epipolar_angle_two_bearings_many(
        b1, b2, rotation, translation
    )
    assert angles_f64.shape == (n, n)
    for i in range(n):
        assert angles_f64[i, i] < 1e-5, (
            f"float64 diagonal [{i},{i}] = {angles_f64[i, i]}, expected < 1e-5"
        )
    for i in range(n):
        for j in range(n):
            if i != j:
                assert angles_f64[i, j] > 1e-5, (
                    f"float64 off-diagonal [{i},{j}] = {angles_f64[i, j]}, expected > 1e-5"
                )

    # Verify with float32 bearings (as used in matching.compute_inliers_bearing_epipolar).
    angles_f32 = pygeometry.epipolar_angle_two_bearings_many(
        b1.astype(np.float32), b2.astype(np.float32), rotation, translation
    )
    assert angles_f32.shape == (n, n)
    for i in range(n):
        assert angles_f32[i, i] < 1e-4, (
            f"float32 diagonal [{i},{i}] = {angles_f32[i, i]}, expected < 1e-4"
        )
    for i in range(n):
        for j in range(n):
            if i != j:
                assert angles_f32[i, j] > 1e-5, (
                    f"float32 off-diagonal [{i},{j}] = {angles_f32[i, j]}, expected > 1e-5"
                )
