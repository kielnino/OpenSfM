# pyre-strict
import logging
import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import opensfm.synthetic_data.synthetic_dataset as sd
import scipy.signal as signal
import scipy.spatial as spatial
from numpy.typing import NDArray
from opensfm import (
    features as oft,
    geo,
    geometry,
    pygeometry,
    pymap,
    reconstruction as rc,
    types,
)
from opensfm.types import Reconstruction


logger: logging.Logger = logging.getLogger(__name__)


def derivative(func: Callable[[float], NDArray], x: float) -> NDArray:
    eps = 1e-10
    d = (func(x + eps) - func(x)) / eps
    d /= np.linalg.norm(d)
    return d


def samples_generator_random_count(count: int) -> NDArray:
    return np.random.rand(count)


def samples_generator_interval(
    length: float, end: float, interval: float, interval_noise: float
) -> NDArray:
    samples = np.linspace(0, end / length, num=int(end / interval))
    samples += np.random.normal(
        0.0, float(interval_noise) / float(length), samples.shape
    )
    return samples


def generate_samples_and_local_frame(
    samples: NDArray, shape: Callable[[float], NDArray]
) -> Tuple[NDArray, NDArray]:
    points = []
    tangents = []
    for i in samples:
        point = shape(i)
        points += [point]
        ex = derivative(shape, i)
        ez = np.array([ex[1], -ex[0]])
        tangents += [np.array([ez, ex])]
    return np.array(points), np.array(tangents)


def generate_samples_shifted(
    samples: NDArray, shape: Callable[[float], NDArray], shift: float
) -> NDArray:
    plane_points = []
    for i in samples:
        point = shape(i)
        tangent = derivative(shape, i)
        tangent = np.array([-tangent[1], tangent[0]])
        point += tangent * (shift / 2)
        plane_points += [point]
    return np.array(plane_points)


def generate_z_plane(
    samples: NDArray, shape: Callable[[float], NDArray], thickness: float
) -> NDArray:
    plane_points = []
    for i in samples:
        point = shape(i)
        tangent = derivative(shape, i)
        tangent = np.array([-tangent[1], tangent[0]])
        shift = tangent * ((np.random.rand() - 0.5) * thickness)
        point += shift
        plane_points += [point]
    plane_points = np.array(plane_points)
    return np.insert(plane_points, 2, values=0, axis=1)


def generate_xy_planes(
    samples: NDArray, shape: Callable[[float], NDArray], z_size: float, y_size: float
) -> NDArray:
    xy1 = generate_samples_shifted(samples, shape, y_size)
    xy2 = generate_samples_shifted(samples, shape, -y_size)
    xy1 = np.insert(xy1, 2, values=np.random.rand(xy1.shape[0]) * z_size, axis=1)
    xy2 = np.insert(xy2, 2, values=np.random.rand(xy2.shape[0]) * z_size, axis=1)
    return np.concatenate((xy1, xy2), axis=0)


def generate_street(
    samples: NDArray, shape: Callable[[float], NDArray], height: float, width: float
) -> Tuple[NDArray, NDArray]:
    walls = generate_xy_planes(samples, shape, height, width)
    floor = generate_z_plane(samples, shape, width)
    return walls, floor


def generate_cameras(
    samples: NDArray, shape: Callable[[float], NDArray], height: float
) -> Tuple[NDArray, NDArray]:
    positions, rotations = generate_samples_and_local_frame(samples, shape)
    positions = np.insert(positions, 2, values=height, axis=1)
    rotations = np.insert(rotations, 2, values=0, axis=2)
    rotations = np.insert(rotations, 1, values=np.array([0, 0, -1]), axis=1)
    return positions, rotations


def line_generator(
    length: float, center_x: float, center_y: float, transpose: bool, point: float
) -> NDArray:
    x = point * length
    if transpose:
        return np.transpose(
            np.array(
                [
                    center_y,
                    x + center_x,
                ]
            )
        )
    else:
        return np.transpose(np.array([x + center_x, center_y]))


def ellipse_generator(x_size: float, y_size: float, point: float) -> NDArray:
    y = np.sin(point * 2 * np.pi) * y_size / 2
    x = np.cos(point * 2 * np.pi) * x_size / 2
    return np.transpose(np.array([x, y]))


def perturb_points(points: NDArray, sigmas: List[float]) -> None:
    eps = 1e-10
    gaussian = np.array([max(s, eps) for s in sigmas])
    for point in points:
        point += np.random.normal(0.0, gaussian, point.shape)


def generate_causal_noise(
    dimensions: int, sigma: float, n: int, scale: float
) -> List[NDArray]:
    dims = [np.arange(-scale, scale) for _ in range(dimensions)]
    mesh = np.meshgrid(*dims)
    dist = np.linalg.norm(mesh, axis=0)
    filter_kernel = np.exp(-(dist**2) / (2 * scale))

    noise = np.random.randn(dimensions, n) * sigma
    return signal.fftconvolve(noise, filter_kernel, mode="same")


def generate_exifs(
    reconstruction: types.Reconstruction,
    reference: geo.TopocentricConverter,
    gps_noise: Union[Dict[str, float], float],
    imu_noise: float,
    causal_gps_noise: bool = False,
) -> Dict[str, Any]:
    """Generate fake exif metadata from the reconstruction."""

    def _gps_dop(shot: pymap.Shot) -> float:
        gps_dop = 15.0
        if isinstance(gps_noise, float):
            gps_dop = gps_noise
        if isinstance(gps_noise, dict):
            gps_dop = gps_noise[shot.camera.id]
        return gps_dop

    exifs = {}
    per_sequence = defaultdict(list)
    for shot_name in sorted(reconstruction.shots.keys()):
        shot = reconstruction.shots[shot_name]
        exif = {}
        exif["width"] = shot.camera.width
        exif["height"] = shot.camera.height
        exif["camera"] = str(shot.camera.id)
        exif["make"] = str(shot.camera.id)

        exif["skey"] = shot.metadata.sequence_key.value
        per_sequence[exif["skey"]].append(shot_name)

        if shot.camera.projection_type in ["perspective", "fisheye"]:
            exif["focal_ratio"] = shot.camera.focal

        exifs[shot_name] = exif

    speed_ms = 10.0
    previous_pose = None
    previous_time = 0
    for rig_instance in sorted(
        reconstruction.rig_instances.values(), key=lambda x: x.id
    ):
        pose = rig_instance.pose.get_origin()
        if previous_pose is not None:
            # pyre-fixme[58]: `+` is not supported for operand types `int` and
            #  `floating[typing.Any]`.
            previous_time += np.linalg.norm(pose - previous_pose) / speed_ms
        previous_pose = pose
        for shot_id in rig_instance.shots:
            exifs[shot_id]["capture_time"] = previous_time

    for sequence_images in per_sequence.values():
        if causal_gps_noise:
            sequence_gps_dop = _gps_dop(reconstruction.shots[sequence_images[0]])
            perturbations_2d = generate_causal_noise(
                2, sequence_gps_dop, len(sequence_images), 2.0
            )
        for i, shot_name in enumerate(sequence_images):
            shot = reconstruction.shots[shot_name]
            exif = exifs[shot_name]

            origin = shot.pose.get_origin()

            if causal_gps_noise:
                # pyre-fixme[61]: `perturbations_2d` is undefined, or not always
                #  defined.
                gps_perturbation = [perturbations_2d[j][i] for j in range(2)] + [0]
            else:
                gps_noise = _gps_dop(shot)
                gps_perturbation = [gps_noise, gps_noise, 0]

            origin = np.array([origin])
            perturb_points(origin, gps_perturbation)
            origin = origin[0]
            _, _, _, comp = rc.shot_lla_and_compass(shot, reference)
            lat, lon, alt = reference.to_lla(*origin)

            exif["gps"] = {}
            exif["gps"]["latitude"] = lat
            exif["gps"]["longitude"] = lon
            exif["gps"]["altitude"] = alt
            exif["gps"]["dop"] = _gps_dop(shot)

            omega, phi, kappa = geometry.opk_from_rotation(
                shot.pose.get_rotation_matrix()
            )
            opk_noise = np.random.normal(0.0, np.full((3), imu_noise), (3))
            exif["opk"] = {}
            exif["opk"]["omega"] = math.degrees(omega) + opk_noise[0]
            exif["opk"]["phi"] = math.degrees(phi) + opk_noise[1]
            exif["opk"]["kappa"] = math.degrees(kappa) + opk_noise[2]

            exif["compass"] = {"angle": comp}

    return exifs


def perturb_rotations(rotations: NDArray, angle_sigma: float) -> None:
    for i in range(len(rotations)):
        rotation = rotations[i]
        rodrigues = cv2.Rodrigues(rotation)[0].ravel()
        angle = np.linalg.norm(rodrigues)
        angle_pertubed = angle + np.random.normal(0.0, angle_sigma)
        rodrigues *= float(angle_pertubed) / float(angle)
        rotations[i] = cv2.Rodrigues(rodrigues)[0]


def add_points_to_reconstruction(
    points: NDArray, color: NDArray, reconstruction: types.Reconstruction
) -> None:
    shift = len(reconstruction.points)
    for i in range(points.shape[0]):
        point = reconstruction.create_point(str(shift + i), points[i, :])
        point.color = color


def add_shots_to_reconstruction(
    shots: List[List[str]],
    positions: List[NDArray],
    rotations: List[NDArray],
    rig_cameras: List[pymap.RigCamera],
    cameras: List[pygeometry.Camera],
    reconstruction: types.Reconstruction,
    sequence_key: str,
) -> None:
    for camera in cameras:
        reconstruction.add_camera(camera)

    rec_rig_cameras = []
    for rig_camera in rig_cameras:
        rec_rig_cameras.append(reconstruction.add_rig_camera(rig_camera))

    for i_shots, position, rotation in zip(shots, positions, rotations):
        instance_id = "_".join([s[0] for s in i_shots])
        rig_instance = reconstruction.add_rig_instance(pymap.RigInstance(instance_id))
        rig_instance.pose = pygeometry.Pose(rotation, -rotation.dot(position))

        for shot, camera in zip(i_shots, cameras):
            shot_id = shot[0]
            rig_camera_id = shot[1]
            shot = reconstruction.create_shot(
                shot_id,
                camera.id,
                pose=None,
                rig_camera_id=rig_camera_id,
                rig_instance_id=instance_id,
            )
            shot.metadata.sequence_key.value = sequence_key


def create_reconstruction(
    points: List[NDArray],
    colors: List[NDArray],
    cameras: List[List[pygeometry.Camera]],
    shot_ids: List[List[str]],
    rig_shots: List[List[List[Tuple[str, str]]]],
    rig_positions: List[NDArray],
    rig_rotations: List[NDArray],
    rig_cameras: List[List[pymap.RigCamera]],
    reference: Optional[geo.TopocentricConverter],
) -> Reconstruction:
    reconstruction = types.Reconstruction()
    if reference is not None:
        reconstruction.reference = reference
    for point, color in zip(points, colors):
        add_points_to_reconstruction(point, color, reconstruction)

    for i, (
        s_rig_shots,
        s_rig_positions,
        s_rig_rotations,
        s_rig_cameras,
        s_cameras,
    ) in enumerate(zip(rig_shots, rig_positions, rig_rotations, rig_cameras, cameras)):
        add_shots_to_reconstruction(
            # pyre-fixme[6]: For 1st argument expected `List[List[str]]` but got
            #  `List[List[Tuple[str, str]]]`.
            s_rig_shots,
            # pyre-fixme[6]: For 2nd argument expected `List[ndarray]` but got
            #  `ndarray`.
            s_rig_positions,
            # pyre-fixme[6]: For 3rd argument expected `List[ndarray]` but got
            #  `ndarray`.
            s_rig_rotations,
            s_rig_cameras,
            s_cameras,
            reconstruction,
            str(f"sequence_{i}"),
        )
    return reconstruction


def generate_track_data(
    reconstruction: types.Reconstruction,
    maximum_depth: float,
    projection_noise: float,
    gcp_noise: Tuple[float, float],
    gcps_count: Optional[int],
    gcp_shift: Optional[NDArray],
    on_disk_features_filename: Optional[str],
) -> Tuple[
    sd.SyntheticFeatures, pymap.TracksManager, Dict[str, pymap.GroundControlPoint]
]:
    """Generate projection data from a reconstruction, considering a maximum
    viewing depth and gaussian noise added to the ideal projections.
    Returns feature/descriptor/color data per shot and a tracks manager object.
    """

    tracks_manager = pymap.TracksManager()

    feature_data_type = np.float32
    desc_size = 128
    non_zeroes = 5

    points_ids = list(reconstruction.points)
    points_coordinates = [p.coordinates for p in reconstruction.points.values()]
    points_colors = [p.color for p in reconstruction.points.values()]

    # generate random descriptors per point
    track_descriptors = []
    for _ in points_coordinates:
        descriptor = np.zeros(desc_size)
        for _ in range(non_zeroes):
            index = np.random.randint(0, desc_size)
            descriptor[index] = np.random.random() * 255
        track_descriptors.append(descriptor.round().astype(feature_data_type))

    # should speed-up projection queries
    points_tree = spatial.cKDTree(points_coordinates)

    start = time.time()
    features = sd.SyntheticFeatures(on_disk_features_filename)
    default_scale = 0.004
    for index, (shot_index, shot) in enumerate(reconstruction.shots.items()):
        # query all closest points
        neighbors = list(
            sorted(points_tree.query_ball_point(shot.pose.get_origin(), maximum_depth))
        )

        # project them
        projections = shot.project_many(
            np.array([points_coordinates[c] for c in neighbors])
        )

        # shot constants
        center = shot.pose.get_origin()
        z_axis = shot.pose.get_rotation_matrix()[2]
        is_panorama = pygeometry.Camera.is_panorama(shot.camera.projection_type)
        perturbation = float(projection_noise) / float(
            max(shot.camera.width, shot.camera.height)
        )
        sigmas = np.array([perturbation, perturbation])

        # pre-generate random perturbations
        perturbations = np.random.normal(0.0, sigmas, (len(projections), 2))

        # run and check valid projections
        projections_inside = []
        descriptors_inside = []
        colors_inside = []
        for i, (p_id, projection) in enumerate(zip(neighbors, projections)):
            if not _is_inside_camera(projection, shot.camera):
                continue

            point = points_coordinates[p_id]
            if not is_panorama and not _is_in_front(point, center, z_axis):
                continue

            # add perturbation
            projection += perturbations[i]

            # push data
            color = points_colors[p_id]
            original_id = points_ids[p_id]
            projections_inside.append([projection[0], projection[1], default_scale])
            descriptors_inside.append(track_descriptors[p_id])
            colors_inside.append(color)
            obs = pymap.Observation(
                projection[0],
                projection[1],
                default_scale,
                color[0],
                color[1],
                color[2],
                len(projections_inside) - 1,
            )
            tracks_manager.add_observation(str(shot_index), str(original_id), obs)
        features[shot_index] = oft.FeaturesData(
            np.array(projections_inside),
            np.array(descriptors_inside),
            np.array(colors_inside),
            None,
        )

        if index % 100 == 0:
            logger.info(
                f"Flushing images # {index} ({(time.time() - start)/(index+1)} sec. per image"
            )
            features.sync()

    gcps = {}
    if gcps_count is not None and gcp_shift is not None:
        all_track_ids = list(tracks_manager.get_track_ids())
        gcps_ids = [
            all_track_ids[i]
            for i in np.random.randint(len(all_track_ids) - 1, size=gcps_count)
        ]

        sigmas_gcp = np.random.normal(
            0.0,
            np.array([gcp_noise[0], gcp_noise[0], gcp_noise[1]]),
            (len(gcps_ids), 3),
        )
        for i, gcp_id in enumerate(gcps_ids):
            point = reconstruction.points[gcp_id]
            gcp = pymap.GroundControlPoint()
            gcp.id = f"gcp-{gcp_id}"
            gcp.survey_point_id = int(gcp_id)
            enu = point.coordinates + gcp_shift + sigmas_gcp[i]
            lat, lon, alt = reconstruction.reference.to_lla(*enu)
            gcp.lla = {"latitude": lat, "longitude": lon, "altitude": alt}
            gcp.has_altitude = True
            for shot_id, obs in tracks_manager.get_track_observations(gcp_id).items():
                o = pymap.GroundControlPointObservation()
                o.shot_id = shot_id
                o.projection = obs.point
                o.uid = obs.id
                gcp.add_observation(o)
            gcps[gcp.id] = gcp

    return features, tracks_manager, gcps


def _is_in_front(point: NDArray, center: NDArray, z_axis: NDArray) -> bool:
    return (
        (point[0] - center[0]) * z_axis[0]
        + (point[1] - center[1]) * z_axis[1]
        + (point[2] - center[2]) * z_axis[2]
    ) > 0


def _is_inside_camera(projection: NDArray, camera: pygeometry.Camera) -> bool:
    w, h = float(camera.width), float(camera.height)
    w2 = float(2 * camera.width)
    h2 = float(2 * camera.height)
    if w > h:
        return (-0.5 < projection[0] < 0.5) and (-h / w2 < projection[1] < h / w2)
    else:
        return (-0.5 < projection[1] < 0.5) and (-w / h2 < projection[0] < w / h2)
