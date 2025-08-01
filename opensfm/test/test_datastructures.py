# pyre-strict
import copy

import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pytest
from opensfm import pygeometry, pymap, types
from opensfm.pymap import RigCamera, RigInstance, Shot
from opensfm.test.utils import (
    assert_cameras_equal,
    assert_maps_equal,
    assert_metadata_equal,
    assert_shots_equal,
)
from opensfm.types import Reconstruction


def _create_reconstruction(
    n_cameras: int = 0,
    n_shots_cam: Optional[Dict[str, int]] = None,
    n_pano_shots_cam: Optional[Dict[str, int]] = None,
    n_points: int = 0,
    dist_to_shots: bool = False,
    dist_to_pano_shots: bool = False,
) -> types.Reconstruction:
    """Creates a reconstruction with n_cameras random cameras and
    shots, where n_shots_cam is a dictionary, containing the
    camera_id and the number of shots.

    Example:
    shot_cams = {"0": 50, "1": 30}
    _create_reconstruction(2, shot_cams)

    Will create a reconstruction with two cameras and 80 shots,
    50 are associated with cam "0" and 30 with cam "1".

    n_points_in_shots is the number of points to create.
    If dist_to_shots, then observations are created and randomly
    distributed to all shots. We pick with the repeat option, thus
    if we have three shots the distribution could be
    something like: [1,2,2], [0,1,2]. We avoid things like [3,3,3]
    """
    if n_shots_cam is None:
        n_shots_cam = {}
    if n_pano_shots_cam is None:
        n_pano_shots_cam = {}

    rec = types.Reconstruction()
    if n_cameras > 0:
        for i in range(n_cameras):
            focal, k1, k2 = np.random.rand(3)
            cam = pygeometry.Camera.create_perspective(focal, k1, k2)
            cam.id = str(i)
            rec.add_camera(cam)

        shot_id = 0
        for cam_id, n_shots in n_shots_cam.items():
            for _ in range(n_shots):
                rec.create_shot(str(shot_id), cam_id)
                shot_id += 1

        shot_id = 0
        for cam_id, n_shots in n_pano_shots_cam.items():
            for _ in range(n_shots):
                rec.create_pano_shot(str(shot_id), cam_id)
                shot_id += 1

    if n_points > 0:
        for i in range(n_points):
            rec.create_point(str(i), np.random.rand(3))

        if dist_to_shots:
            n_shots = len(rec.shots)
            for pt in rec.points.values():
                choice = set(np.random.choice(n_shots, n_shots))
                if len(choice) > 1:
                    for ch in choice:
                        # create a new observation
                        obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, int(pt.id))
                        shot = rec.shots[str(ch)]
                        rec.add_observation(shot, pt, obs)
        # TODO: If required, we have to do the same for pano shots
    return rec


"""
Camera Tests
"""


def test_create_cameras() -> None:
    n_cameras = 100
    rec = types.Reconstruction()

    for cam_id in range(0, n_cameras):
        focal, k1, k2 = np.random.rand(3)
        cam = pygeometry.Camera.create_perspective(focal, k1, k2)
        cam.id = str(cam_id)
        # create the camera within the reconstruction
        map_cam = rec.add_camera(cam)
        assert_cameras_equal(cam, map_cam)
        # Check that the cameras are different
        assert cam is not map_cam
        # Check the getters
        assert map_cam is rec.get_camera(str(cam_id))
        assert map_cam is rec.cameras[str(cam_id)]

    assert len(rec.cameras) == n_cameras


def test_camera_iterators() -> None:
    n_cameras = 100
    rec = _create_reconstruction(n_cameras)

    # Key iterator
    visited_cams = set()
    for cam_id in rec.cameras:
        visited_cams.add(cam_id)
    assert len(visited_cams) == n_cameras

    for idx in range(0, n_cameras):
        assert str(idx) in visited_cams

    # value iterator
    visited_cams = set()
    for cam in rec.cameras.values():
        visited_cams.add(cam.id)
        focal = np.random.rand(1)
        cam.focal = focal
        assert rec.cameras[cam.id].focal == focal
        assert cam is rec.cameras[cam.id]

    assert len(visited_cams) == n_cameras

    # item iterator
    for idx in range(0, n_cameras):
        assert str(idx) in visited_cams

    for cam_id, cam in rec.cameras.items():
        assert cam_id == cam.id
        focal = np.random.rand(1)
        cam.focal = focal
        assert rec.cameras[cam.id].focal == focal
        assert cam is rec.cameras[cam.id]


def _check_common_cam_properties(
    cam1: pygeometry.Camera, cam2: pygeometry.Camera
) -> None:
    assert cam1.id == cam2.id
    assert cam1.width == cam2.width
    assert cam1.height == cam2.height
    assert cam1.projection_type == cam2.projection_type


def test_brown_camera() -> None:
    rec = types.Reconstruction()
    focal_x = 0.6
    focal_y = 0.7
    c_x = 0.1
    c_y = -0.05
    k1 = -0.1
    k2 = 0.01
    p1 = 0.001
    p2 = 0.002
    k3 = 0.01
    cam_cpp = pygeometry.Camera.create_brown(
        focal_x, focal_y / focal_x, np.array([c_x, c_y]), np.array([k1, k2, k3, p1, p2])
    )
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2 and cam_cpp.k3 == c.k3
    assert cam_cpp.p2 == c.p2 and cam_cpp.p1 == c.p1
    assert np.allclose(cam_cpp.principal_point, c.principal_point)
    assert len(c.distortion) == 5
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio


def test_fisheye_camera() -> None:
    rec = types.Reconstruction()
    focal = 0.6
    k1 = -0.1
    k2 = 0.01
    cam_cpp = pygeometry.Camera.create_fisheye(focal, k1, k2)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert len(c.distortion) == 2
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal


def test_fisheye_opencv_camera() -> None:
    rec = types.Reconstruction()
    focal = 0.6
    aspect_ratio = 0.7
    ppoint = np.array([0.51, 0.52])
    dist = np.array([-0.1, 0.09, 0.08, 0.01])
    cam_cpp = pygeometry.Camera.create_fisheye_opencv(focal, aspect_ratio, ppoint, dist)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert cam_cpp.k3 == c.k3 and cam_cpp.k4 == c.k4
    assert len(dist) == len(c.distortion)
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio


def test_fisheye62_camera() -> None:
    rec = types.Reconstruction()
    focal = 0.6
    aspect_ratio = 0.7
    ppoint = np.array([0.51, 0.52])
    dist = np.array([-0.1, 0.09, 0.08, 0.01, 0.02, 0.05, 0.1, 0.2])  # [k1-k6, p1, p2]
    cam_cpp = pygeometry.Camera.create_fisheye62(focal, aspect_ratio, ppoint, dist)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert cam_cpp.k3 == c.k3 and cam_cpp.k4 == c.k4
    assert cam_cpp.k5 == c.k5 and cam_cpp.k6 == c.k6
    assert cam_cpp.p1 == c.p1 and cam_cpp.p2 == c.p2
    assert len(dist) == len(c.distortion)
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio


def test_fisheye624_camera() -> None:
    rec = types.Reconstruction()
    focal = 0.6
    aspect_ratio = 0.7
    ppoint = np.array([0.51, 0.52])
    dist = np.array(
        [-0.1, 0.09, 0.08, 0.01, 0.02, 0.05, 0.1, 0.2, 0.01, -0.003, 0.005, -0.007]
    )  # [k1-k6, p1, p2, s0-s3]
    cam_cpp = pygeometry.Camera.create_fisheye624(focal, aspect_ratio, ppoint, dist)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert cam_cpp.k3 == c.k3 and cam_cpp.k4 == c.k4
    assert cam_cpp.k5 == c.k5 and cam_cpp.k6 == c.k6
    assert cam_cpp.p1 == c.p1 and cam_cpp.p2 == c.p2
    assert cam_cpp.s0 == c.s0 and cam_cpp.s1 == c.s1
    assert cam_cpp.s2 == c.s2 and cam_cpp.s3 == c.s3
    assert len(dist) == len(c.distortion)
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio


def test_dual_camera() -> None:
    rec = types.Reconstruction()
    focal = 0.6
    k1 = -0.1
    k2 = 0.01
    transition = 0.5
    cam_cpp = pygeometry.Camera.create_dual(transition, focal, k1, k2)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert len(c.distortion) == 2
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.transition == c.transition


def test_perspective_camera() -> None:
    rec = types.Reconstruction()
    focal = 0.6
    k1 = -0.1
    k2 = 0.01
    cam_cpp = pygeometry.Camera.create_perspective(focal, k1, k2)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

    # The specific parameters
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert len(c.distortion) == 2
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal


def test_spherical_camera() -> None:
    rec = types.Reconstruction()
    cam_cpp = pygeometry.Camera.create_spherical()
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = "cam"
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)


# Test Metadata
def _help_measurement_test(
    measurement: object, attr: str, val: Union[float, str]
) -> None:
    # Test metadata's has_value properties
    assert getattr(measurement, attr).has_value is False
    getattr(measurement, attr).value = val
    if np.shape(val) == ():  # just a value
        assert getattr(measurement, attr).value == val
    else:
        assert np.allclose(getattr(measurement, attr).value, val)
    # Test metadata's has_value properties!
    assert getattr(measurement, attr).has_value is True
    # Test reset
    getattr(measurement, attr).reset()
    assert getattr(measurement, attr).has_value is False


def test_shot_measurement_setter_and_getter() -> None:
    m1 = pymap.ShotMeasurements()
    # Test basic functionality
    _help_measurement_test(m1, "capture_time", np.random.rand(1))
    _help_measurement_test(m1, "gps_position", np.random.rand(3))
    _help_measurement_test(m1, "gps_accuracy", np.random.rand(1))
    _help_measurement_test(m1, "compass_accuracy", np.random.rand(1))
    _help_measurement_test(m1, "compass_angle", np.random.rand(1))
    _help_measurement_test(m1, "opk_accuracy", np.random.rand(1))
    _help_measurement_test(m1, "opk_angles", np.random.rand(3))
    _help_measurement_test(m1, "gravity_down", np.random.rand(3))
    _help_measurement_test(m1, "orientation", random.randint(0, 100))
    _help_measurement_test(m1, "sequence_key", "key_test")


def _helper_populate_metadata(m: pymap.ShotMeasurements) -> None:
    m.capture_time.value = np.random.rand(1)
    m.gps_position.value = np.random.rand(3)
    m.gps_accuracy.value = np.random.rand(1)
    m.compass_accuracy.value = np.random.rand(1)
    m.compass_angle.value = np.random.rand(1)
    m.opk_accuracy.value = np.random.rand(1)
    m.opk_angles.value = np.random.rand(3)
    m.gravity_down.value = np.random.rand(3)
    m.orientation.value = random.randint(0, 100)
    m.sequence_key.value = "sequence_key"


def test_shot_measurement_set() -> None:
    m1 = pymap.ShotMeasurements()
    _helper_populate_metadata(m1)
    m2 = pymap.ShotMeasurements()
    # Test setting metadata with other metadata
    m2.set(m1)
    # Check that m2 has the same values as m1
    assert_metadata_equal(m1, m2)
    m3 = pymap.ShotMeasurements()
    m1.set(m3)
    # Now m1 should be completely reset
    assert_metadata_equal(m1, m3)


def test_shot_create() -> None:
    # Given some created shot
    rec = _create_reconstruction(2)
    shot1 = rec.create_shot("shot0", "0")

    # When getting it, it should have some properties
    assert shot1.id == "shot0"
    assert shot1.camera.id == "0"
    assert len(rec.shots) == 1


def test_shot_create_existing() -> None:
    # Given some created shot
    rec = _create_reconstruction(2)
    rec.create_shot("shot0", "0")

    # When re-adding the same shot, it should throw
    with pytest.raises(RuntimeError):
        rec.create_shot("shot0", "0")
        rec.create_shot("shot0", "1")


def test_shot_create_more() -> None:
    # Given some created shot
    rec = _create_reconstruction(2)
    rec.create_shot("shot0", "0")

    # When we create more new shots
    n_shots = 10
    for i in range(1, n_shots):
        rec.create_shot("shot" + str(i), "0")

    # Then we should have more
    assert len(rec.shots) == n_shots


def test_shot_delete_non_existing() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(2)
    rec.create_shot("shot0", "0")

    # When deleting non-existing shot
    # It should throw
    with pytest.raises(RuntimeError):
        rec.remove_shot("abcde")


def test_shot_delete_existing() -> None:
    # Given some created reconstruction
    n_shots = 10
    rec = _create_reconstruction(1, {"0": n_shots})

    # When deleting existing shot
    del_shots = np.random.choice(n_shots, int(n_shots / 2), replace=False)
    for i in del_shots:
        rec.remove_shot(str(i))

    # Then we should have the expected count of shots remaining
    assert len(rec.shots) == n_shots - len(del_shots)


def test_shot_get() -> None:
    # Given some created shot
    rec = _create_reconstruction(1)
    shot_id = "shot0"
    shot1 = rec.create_shot(shot_id, "0")

    # We should get it
    assert shot1 is rec.get_shot(shot_id)
    assert shot1 is rec.shots[shot_id]


def test_shot_pose_set() -> None:
    # Given some created shot
    rec = _create_reconstruction(1)
    shot_id = "shot0"
    shot = rec.create_shot(shot_id, "0")

    origin = np.array([1, 2, 3])
    shot.pose.set_origin(origin)
    assert np.allclose(origin, shot.pose.get_origin())


def test_shot_get_non_existing() -> None:
    # Given some created shot
    rec = _create_reconstruction(1)
    shot_id = "shot0"
    shot1 = rec.create_shot(shot_id, "0")

    # When getting a non_existing one, it should throw
    with pytest.raises(RuntimeError):
        assert shot1 is rec.get_shot("toto")
    with pytest.raises(RuntimeError):
        assert shot1 is rec.shots["toto"]


def test_pano_shot_get() -> None:
    # Given some created pano shot
    rec = _create_reconstruction(1)
    shot_id = "shot0"
    shot1 = rec.create_pano_shot(shot_id, "0")

    # We should get it
    assert shot1 is rec.pano_shots[shot_id]
    assert shot1 is rec.get_pano_shot(shot_id)


def test_pano_shot_get_non_existing() -> None:
    # Given some created pano shot
    rec = _create_reconstruction(1)
    shot_id = "shot0"
    shot1 = rec.create_shot(shot_id, "0")

    # When getting a non_existing one, it should throw
    with pytest.raises(RuntimeError):
        assert shot1 is rec.get_shot("toto")
    with pytest.raises(RuntimeError):
        assert shot1 is rec.shots["toto"]


def test_pano_shot_create() -> None:
    # Given some created shot
    rec = _create_reconstruction(2)
    shot1 = rec.create_pano_shot("shot0", "0")

    # When getting it, it should have some properties
    assert shot1.id == "shot0"
    assert shot1.camera.id == "0"
    assert len(rec.pano_shots) == 1


def test_pano_shot_create_existing() -> None:
    # Given some created pano shot
    rec = _create_reconstruction(2)
    rec.create_pano_shot("shot0", "0")

    n_shots = 10
    # When re-adding the same pano shot
    for _ in range(n_shots):
        # It should throw
        with pytest.raises(RuntimeError):
            rec.create_pano_shot("shot0", "0")
            rec.create_pano_shot("shot0", "1")


def test_pano_shot_create_more() -> None:
    # Given some created pano shot
    rec = _create_reconstruction(2)
    rec.create_pano_shot("shot0", "0")

    # When we create more new pano shots
    n_shots = 10
    for i in range(1, n_shots):
        rec.create_pano_shot("shot" + str(i), "0")

    # Then we should have more
    assert len(rec.pano_shots) == n_shots


def test_pano_shot_delete_non_existing() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(2)
    rec.create_pano_shot("shot0", "0")

    # When deleting non-existing shot
    # It should throw
    with pytest.raises(RuntimeError):
        rec.remove_pano_shot("abcde")


def test_pano_shot_delete_existing() -> None:
    # Given some created reconstruction
    n_shots = 10
    rec = _create_reconstruction(2)
    rec = _create_reconstruction(1, n_pano_shots_cam={"0": n_shots})

    # When deleting existing pano shot
    n_shots = 10
    del_shots = np.random.choice(n_shots, int(n_shots / 2), replace=False)
    for i in del_shots:
        rec.remove_pano_shot(str(i))

    # Then we should have the expected count of shots remaining
    assert len(rec.pano_shots) == n_shots - len(del_shots)


def test_shot_merge_cc() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(1, {"0": 2})
    map_shot1 = rec.shots["0"]

    # When setting some merge_cc
    map_shot1.merge_cc = 10

    # Then we should have it set
    assert map_shot1.merge_cc == 10


def test_shot_covariance() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(1, {"0": 2})
    map_shot1 = rec.shots["0"]

    # When setting some covariance
    map_shot1.covariance = np.diag([1, 2, 3])

    # Then we should have it set
    assert np.allclose(map_shot1.covariance, np.diag([1, 2, 3]))


def test_shot_covariance_different() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(1, {"0": 2})
    map_shot1 = rec.shots["0"]
    map_shot2 = rec.shots["1"]

    # When setting some covariance
    map_shot1.covariance = np.diag([1, 2, 3])
    map_shot2.covariance = np.diag([2, 2, 2])

    # Then they are different objects
    assert map_shot2.covariance is not map_shot1.covariance


def test_shot_create_remove_create() -> None:
    # Given some created reconstruction
    n_shots = 10
    rec = _create_reconstruction(1, {"0": n_shots})

    # When we remove one shot
    rec.remove_shot("0")

    # Then we have one shot less
    assert len(rec.shots) == n_shots - 1

    # When we re-create it
    rec.create_shot("0", "0")

    # Then we have the initial count
    assert len(rec.shots) == n_shots


def test_pano_shot_create_remove_create() -> None:
    # Given some created reconstruction
    n_shots = 10
    rec = _create_reconstruction(1, n_pano_shots_cam={"0": n_shots})

    # When we remove one pano shot
    rec.remove_pano_shot("0")

    # Then we have one pano shot less
    assert len(rec.pano_shots) == n_shots - 1

    # When we re-create it
    rec.create_pano_shot("0", "0")

    # Then we have the initial count
    assert len(rec.pano_shots) == n_shots


def _create_rig_camera() -> RigCamera:
    rig_camera = pymap.RigCamera()
    rig_camera.id = "rig_camera"
    rig_camera.pose = pygeometry.Pose(
        np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])
    )
    return rig_camera


def _create_rig_instance() -> Tuple[Reconstruction, RigInstance, Shot]:
    rec = _create_reconstruction(1, {"0": 2})
    rig_camera = rec.add_rig_camera(_create_rig_camera())
    rig_instance = pymap.RigInstance("1")
    shot = pymap.Shot(
        "0",
        pygeometry.Camera.create_spherical(),
        pygeometry.Pose(),
    )
    rig_instance.add_shot(rig_camera, shot)
    return rec, rig_instance, shot


def test_rig_camera_create() -> None:
    rec = _create_reconstruction(1, {"0": 2})
    rec.add_rig_camera(_create_rig_camera())

    # we should have default-per-camera rig and the created rig camera
    assert "0" in rec.rig_cameras.keys()
    assert "rig_camera" in rec.rig_cameras.keys()


def test_rig_instance() -> None:
    _, rig_instance, _ = _create_rig_instance()
    assert list(rig_instance.keys()) == ["0"]


def test_rig_instance_create_default() -> None:
    # one default rig instance per shot
    rec, rig_instance, _ = _create_rig_instance()

    assert len(rec.rig_instances) == 2
    assert dict(rec.rig_instances["0"].camera_ids.items()) == {"0": "0"}
    assert list(rec.rig_instances["0"].shots.keys()) == ["0"]
    assert dict(rec.rig_instances["1"].camera_ids.items()) == {"1": "0"}
    assert list(rec.rig_instances["1"].shots.keys()) == ["1"]


def test_rig_instance_create_add_existing() -> None:
    rec, rig_instance, _ = _create_rig_instance()
    with pytest.raises(RuntimeError):
        rec.add_rig_instance(rig_instance)


def test_rig_instance_remove_shot() -> None:
    rec, _, shot = _create_rig_instance()
    rec.remove_shot(shot.id)
    assert len(rec.rig_instances["0"].shots) == 0


def test_rig_shot_modify_pose_raise() -> None:
    _, rig_instance, shot = _create_rig_instance()
    with pytest.raises(RuntimeError):
        shot.pose.set_origin(np.array([1, 2, 3]))


def test_rig_shot_modify_pose_succeed() -> None:
    _, rig_instance, shot = _create_rig_instance()
    next(iter(rig_instance.rig_cameras.values())).pose = pygeometry.Pose()
    shot.pose.set_origin(np.array([1, 2, 3]))


def test_rig_shot_set_pose() -> None:
    _, rig_instance, shot = _create_rig_instance()
    with pytest.raises(RuntimeError):
        shot.pose = pygeometry.Pose()


def test_add_shot_from_shot_correct_value() -> None:
    # Given some created reconstruction (rec) ...
    n_shots = 5
    rec = _create_reconstruction(1, n_shots_cam={"0": n_shots})
    shot1 = rec.shots["0"]
    _helper_populate_metadata(shot1.metadata)

    # .. and given another one (new)
    rec_new = _create_reconstruction(1)

    # When adding 2 shot of rec to new
    rec_new.add_shot(rec.shots["0"])
    rec_new.add_shot(rec.shots["1"])

    # Then new has two shots ...
    assert len(rec_new.shots) == 2

    # ... and new's shots values should be the same as rec's ones'
    for k in rec_new.shots.keys():
        assert_shots_equal(rec.shots[k], rec_new.shots[k])


def test_shot_metadata_different() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(1, n_shots_cam={"0": 2})
    shot1 = rec.shots["0"]
    shot2 = rec.shots["1"]
    _helper_populate_metadata(shot1.metadata)

    # When getting their metadata object, they should be different
    assert shot1.metadata is not shot2.metadata


def test_shot_metadata_assign_equal() -> None:
    # Given some created reconstruction
    rec = _create_reconstruction(1, n_shots_cam={"0": 2})
    shot1 = rec.shots["0"]
    shot2 = rec.shots["1"]
    _helper_populate_metadata(shot1.metadata)

    # When assigning their metadata to be equal
    shot2.metadata = shot1.metadata

    # Their object are different ...
    assert shot1.metadata is not shot2.metadata

    # ... but their values are equal
    assert_metadata_equal(shot1.metadata, shot2.metadata)


def test_add_pano_shot_from_shot_correct_value() -> None:
    # Given some created reconstruction (rec) ...
    n_shots = 5
    rec = _create_reconstruction(1, n_pano_shots_cam={"0": n_shots})
    shot1 = rec.pano_shots["0"]
    _helper_populate_metadata(shot1.metadata)

    # .. and given another one (new)
    rec_new = _create_reconstruction(1)

    # When adding 2 pano shot of rec to new
    rec_new.add_pano_shot(rec.pano_shots["0"])
    rec_new.add_pano_shot(rec.pano_shots["1"])

    # Then new's shots values should be the same as rec's ones'
    for k in rec_new.shots.keys():
        assert_shots_equal(rec.pano_shots[k], rec_new.pano_shots[k])


def test_single_point_create() -> None:
    # Given a created point
    rec = types.Reconstruction()
    pt = rec.create_point("0")

    # It should be there
    assert pt.id == "0"
    assert len(rec.points) == 1


def test_single_point_get_existing() -> None:
    # Given a created point
    rec = types.Reconstruction()
    pt = rec.create_point("0")

    # When we get it, we have it (!)
    assert pt == rec.points["0"] and pt == rec.get_point("0")


def test_single_point_get_non_existing() -> None:
    # Given a created point
    rec = types.Reconstruction()
    rec.create_point("0")

    # When we get a non existing one
    with pytest.raises(RuntimeError):
        # It should throw
        rec.get_point("toto")


def test_single_point_coordinates() -> None:
    # Given a created point
    rec = types.Reconstruction()
    pt = rec.create_point("0")

    # When assigning coordinates
    coord = np.random.rand(3)
    pt.coordinates = coord

    # They should be set
    assert np.allclose(pt.coordinates, coord)


def test_single_point_color() -> None:
    # Given a created point
    rec = types.Reconstruction()
    pt = rec.create_point("0")

    # When assigning color
    color = np.random.randint(low=0, high=255, size=(3,))
    pt.color = color

    # It should be set
    assert np.allclose(pt.color, color)


def test_point_add_from_point() -> None:
    # Given some created reconstruction (rec) ...
    rec = types.Reconstruction()

    # ... and some other one (rec2) with some point
    rec2 = types.Reconstruction()
    coord2 = np.random.rand(3)
    pt2 = rec2.create_point("1", coord2)

    # When adding rec2 point to rec
    pt2_1 = rec.add_point(pt2)

    # Then rec should have it ...
    assert len(rec.points) == 1

    # ... as a different object
    assert pt2 is not pt2_1
    assert "1" == pt2_1.id

    # ... and with correct values
    assert pt2_1 == rec.points["1"]
    assert np.allclose(pt2_1.coordinates, coord2)


def test_point_reproj_errors_assign() -> None:
    # Given some created point
    rec = _create_reconstruction(n_points=1)
    pt = rec.points["0"]

    # When assigning reprojections errors
    reproj_errors = dict({"shot1": np.random.rand(2), "shot2": np.random.rand(2)})
    pt.reprojection_errors = reproj_errors

    # They should be correct
    for k in reproj_errors.keys():
        assert np.allclose(pt.reprojection_errors[k], reproj_errors[k])


def test_point_delete_non_existing() -> None:
    # Given some created points
    n_points = 100
    rec = _create_reconstruction(n_points=n_points)

    # When we delete a non-existing one
    with pytest.raises(RuntimeError):
        # It should throw
        rec.remove_point("abcdef")


def test_point_delete_existing() -> None:
    # Given some created points
    n_points = 100
    rec = _create_reconstruction(n_points=n_points)

    # When we delete all of them
    del_list = list(rec.points.keys())
    for k in del_list:
        rec.remove_point(k)

    # Then there's none
    assert len(rec.points) == 0


def test_point_delete_existing_assign_empty() -> None:
    # Given some created points
    n_points = 100
    rec = _create_reconstruction(n_points=n_points)

    # When we delete all of them by assigning the empty dict
    rec.points = {}
    assert len(rec.points) == 0


def test_single_observation() -> None:
    # Given a 1-camera, 1-point reconstruction
    rec = _create_reconstruction(1, n_shots_cam={"0": 1}, n_points=1)

    # When we add an observation to it
    obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, 100, 2, 5)
    rec.add_observation("0", "0", obs)
    shot = rec.shots["0"]
    pt = rec.points["0"]

    # Then it has one observation ...
    observations = pt.get_observations()
    assert len(observations) == 1
    assert pt.number_of_observations() == 1

    # ... and the corresponding observation object
    obs = shot.get_landmark_observation(pt)
    assert obs is not None


def test_single_observation_delete() -> None:
    # Given a 1-camera, 1-point reconstruction and corresponding observation
    rec = _create_reconstruction(1, n_shots_cam={"0": 1}, n_points=1)
    obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, 100)
    rec.add_observation("0", "0", obs)
    shot = rec.shots["0"]
    pt = rec.points["0"]

    # When we remove it
    rec.remove_observation(shot.id, pt.id)

    # Then there's none
    observations = pt.get_observations()
    assert len(observations) == 0
    assert pt.number_of_observations() == 0


def test_many_observations_delete() -> None:
    # Given a map with 10 shots, 1000 landmarks ...
    m = pymap.Map()
    n_cams = 2
    n_shots = 10
    n_landmarks = 1000
    for cam_id in range(n_cams):
        cam = pygeometry.Camera.create_perspective(0.5, 0, 0)
        cam.id = "cam" + str(cam_id)
        m.create_camera(cam)
        m.create_rig_camera(pymap.RigCamera(pygeometry.Pose(), cam.id))

    for shot_id in range(n_shots):
        cam_id = "cam" + str(int(np.random.rand(1) * 10 % n_cams))
        shot_id = str(shot_id)
        m.create_rig_instance(shot_id)
        m.create_shot(shot_id, cam_id, cam_id, shot_id, pygeometry.Pose())

    for point_id in range(n_landmarks):
        m.create_landmark(str(point_id), np.random.rand(3))

    # ... and random connections (observations) between shots and points
    n_total_obs = 0
    for lm in m.get_landmarks().values():
        n_obs = 0
        for shot in m.get_shots().values():
            # create a new observation
            obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, int(lm.id))
            m.add_observation(shot, lm, obs)
            n_obs += 1
            n_total_obs += 1

    # (we expect it to be created correctly)
    for lm in m.get_landmarks().values():
        n_total_obs -= lm.number_of_observations()
    assert n_total_obs == 0

    # and when we clear all the observations
    m.clear_observations_and_landmarks()


def test_clean_landmarks_with_min_observations() -> None:
    m = pymap.Map()
    n_cams = 2
    n_shots = 2
    n_landmarks = 10
    for cam_id in range(n_cams):
        cam = pygeometry.Camera.create_perspective(0.5, 0, 0)
        cam.id = "cam" + str(cam_id)
        m.create_camera(cam)
        m.create_rig_camera(pymap.RigCamera(pygeometry.Pose(), cam.id))

    for shot_id in range(n_shots):
        cam_id = "cam" + str(int(np.random.rand(1) * 10 % n_cams))
        m.create_rig_instance(str(shot_id))
        m.create_shot(str(shot_id), cam_id, cam_id, str(shot_id), pygeometry.Pose())

    for point_id in range(n_landmarks):
        m.create_landmark(str(point_id), np.random.rand(3))

    for point_id in range(int(n_landmarks / 2)):
        for shot in m.get_shots().values():
            # create a new observation
            obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, point_id)
            m.add_observation(shot, m.get_landmark(str(point_id)), obs)

    for point_id in range(int(n_landmarks / 2), n_landmarks):
        shot = m.get_shot("0")
        # create a new observation
        obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, point_id)
        m.add_observation(shot, m.get_landmark(str(point_id)), obs)

    m.clean_landmarks_below_min_observations(n_shots)

    assert len(m.get_landmarks()) == int(n_landmarks / 2)
    m.clean_landmarks_below_min_observations(n_shots + 1)
    assert len(m.get_landmarks()) == 0


def test_camera_deepcopy() -> None:
    # Given a camera
    cam1 = pygeometry.Camera.create_perspective(0.5, 0, 0)

    # When we deepcopy it
    cam2 = copy.deepcopy(cam1)

    # Then it has the correct focal
    assert cam1.focal == cam2.focal


def test_camera_deepcopy_assign() -> None:
    # Given a camera
    cam1 = pygeometry.Camera.create_perspective(0.5, 0, 0)

    # When we deepcopy'n assign it
    cam2 = copy.deepcopy(cam1)
    cam2.focal = 0.7

    # Then it has a different value from the original
    assert cam1.focal != cam2.focal


def test_observation_shot_removal() -> None:
    # Given a reconstruction with 2 shots
    rec = _create_reconstruction(
        n_cameras=2, n_shots_cam={"0": 1, "1": 1}, n_points=200, dist_to_shots=True
    )

    # When removing one of them
    rec.remove_shot("0")

    # All the points have only one at most observation each ...
    for p in rec.points.values():
        assert len(p.get_observations()) <= 1

    # ... and when removing the remaining one ...
    rec.remove_shot("1")

    # Thers' none
    for p in rec.points.values():
        assert len(p.get_observations()) == 0


def test_rec_deepcopy() -> None:
    # Given a reconstruction with everything (shots, pano shots, metadata)
    rec = _create_reconstruction(
        n_cameras=2,
        n_shots_cam={"0": 50, "1": 40},
        n_pano_shots_cam={"0": 20, "1": 30},
        n_points=200,
        dist_to_shots=True,
    )
    for shot in rec.shots.values():
        _helper_populate_metadata(shot.metadata)
    for shot in rec.pano_shots.values():
        _helper_populate_metadata(shot.metadata)

    # When we deep-copy it
    # pyre-fixme[6]: For 2nd param expected `Optional[Dict[int, typing.Any]]` but
    #  got `Dict[str, bool]`.
    rec2 = copy.deepcopy(rec, {"copy_observations": True})

    # It has the expected count of data
    assert len(rec2.cameras) == 2
    assert len(rec2.shots) == 90
    assert len(rec2.pano_shots) == 50
    assert len(rec2.points) == 200

    assert_maps_equal(rec.map, rec2.map)


def test_gcp() -> None:
    gcp = []
    for i in range(0, 10):
        p = pymap.GroundControlPoint()
        p.id = "p" + str(i)
        o1 = pymap.GroundControlPointObservation()
        o1.shot_id = "p1"
        o2 = pymap.GroundControlPointObservation()
        o2.shot_id = "p2"
        obs = [o1, o2]
        p.observations = obs
        gcp.append(p)
        assert p.observations[0].shot_id == "p1"
        assert p.observations[1].shot_id == "p2"
        p.add_observation(o2)
        p.add_observation(o2)
        assert len(p.observations) == 4
    for pt in gcp:
        assert pt.observations[0].shot_id == "p1"
        assert pt.observations[1].shot_id == "p2"


def test_add_correspondences_from_tracks_manager() -> None:
    n_shots = 3
    rec = _create_reconstruction(
        n_cameras=1,
        n_shots_cam={"0": n_shots},
        n_points=10,
    )
    # create tracks manager
    tm = pymap.TracksManager()
    # add observations for 3 tracks
    # One shot and one landmark are not in the reconstruction
    for track_id in ["0", "1", "100"]:
        for shot_id in range(n_shots + 1):
            obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, 100)
            tm.add_observation(str(shot_id), track_id, obs)

    # add a shot that is NOT in the tracks manager
    rec.create_shot(str(n_shots + 5), next(iter(rec.cameras)))

    rec.add_correspondences_from_tracks_manager(tm)

    # make sure to have the observations for []
    assert "100" not in rec.points

    for track_id in ["0", "1"]:
        pt = rec.points[track_id]
        observations = pt.get_observations()
        assert len(observations) == n_shots
