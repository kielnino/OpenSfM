# pyre-strict
"""Incremental reconstruction pipeline"""

import datetime
import enum
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from numpy import typing as npt
from numpy.typing import NDArray
from opensfm import (
    log,
    matching,
    multiview,
    pybundle,
    pygeometry,
    pymap,
    pysfm,
    reconstruction_helpers as helpers,
    rig,
    tracking,
    types,
)
from opensfm.align import align_reconstruction, apply_similarity
from opensfm.context import current_memory_usage, parallel_map
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


class ReconstructionAlgorithm(str, enum.Enum):
    INCREMENTAL = "incremental"
    TRIANGULATION = "triangulation"


def _get_camera_from_bundle(
    ba: pybundle.BundleAdjuster, camera: pygeometry.Camera
) -> None:
    """Read camera parameters from a bundle adjustment problem."""
    c = ba.get_camera(camera.id)
    for k, v in c.get_parameters_map().items():
        camera.set_parameter_value(k, v)


def log_bundle_stats(bundle_type: str, bundle_report: Dict[str, Any]) -> None:
    times = bundle_report["wall_times"]
    time_secs = times["run"] + times["setup"] + times["teardown"]
    num_images, num_points, num_reprojections = (
        bundle_report["num_images"],
        bundle_report["num_points"],
        bundle_report["num_reprojections"],
    )

    msg = f"Ran {bundle_type} bundle in {time_secs:.2f} secs."
    if num_points > 0:
        msg += f"with {num_images}/{num_points}/{num_reprojections} ({num_reprojections/num_points:.2f}) "
        msg += "shots/points/proj. (avg. length)"

    logger.info(msg)


def bundle(
    reconstruction: types.Reconstruction,
    camera_priors: Dict[str, pygeometry.Camera],
    rig_camera_priors: Dict[str, pymap.RigCamera],
    gcp: Optional[List[pymap.GroundControlPoint]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Bundle adjust a reconstruction."""
    report = pysfm.BAHelpers.bundle(
        reconstruction.map,
        dict(camera_priors),
        dict(rig_camera_priors),
        gcp if gcp is not None else [],
        config,
    )
    log_bundle_stats("GLOBAL", report)
    logger.debug(report["brief_report"])
    return report


def bundle_shot_poses(
    reconstruction: types.Reconstruction,
    shot_ids: Set[str],
    camera_priors: Dict[str, pygeometry.Camera],
    rig_camera_priors: Dict[str, pymap.RigCamera],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Bundle adjust a set of shots poses."""
    report = pysfm.BAHelpers.bundle_shot_poses(
        reconstruction.map,
        shot_ids,
        dict(camera_priors),
        dict(rig_camera_priors),
        config,
    )
    return report


def bundle_local(
    reconstruction: types.Reconstruction,
    camera_priors: Dict[str, pygeometry.Camera],
    rig_camera_priors: Dict[str, pymap.RigCamera],
    gcp: Optional[List[pymap.GroundControlPoint]],
    central_shot_id: str,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[int]]:
    """Bundle adjust the local neighborhood of a shot."""
    pt_ids, report = pysfm.BAHelpers.bundle_local(
        reconstruction.map,
        dict(camera_priors),
        dict(rig_camera_priors),
        gcp if gcp is not None else [],
        central_shot_id,
        config,
    )
    log_bundle_stats("LOCAL", report)
    logger.debug(report["brief_report"])
    return pt_ids, report


def shot_neighborhood(
    reconstruction: types.Reconstruction,
    central_shot_id: str,
    radius: int,
    min_common_points: int,
    max_interior_size: int,
) -> Tuple[Set[str], Set[str]]:
    """Reconstructed shots near a given shot.

    Returns:
        a tuple with interior and boundary:
        - interior: the list of shots at distance smaller than radius
        - boundary: shots sharing at least on point with the interior

    Central shot is at distance 0.  Shots at distance n + 1 share at least
    min_common_points points with shots at distance n.
    """
    max_boundary_size = 1000000
    interior = {central_shot_id}
    for _distance in range(1, radius):
        remaining = max_interior_size - len(interior)
        if remaining <= 0:
            break
        neighbors = direct_shot_neighbors(
            reconstruction, interior, min_common_points, remaining
        )
        interior.update(neighbors)
    boundary = direct_shot_neighbors(reconstruction, interior, 1, max_boundary_size)
    return interior, boundary


def direct_shot_neighbors(
    reconstruction: types.Reconstruction,
    shot_ids: Set[str],
    min_common_points: int,
    max_neighbors: int,
) -> Set[str]:
    """Reconstructed shots sharing reconstructed points with a shot set."""
    points = set()
    for shot_id in shot_ids:
        shot = reconstruction.shots[shot_id]
        valid_landmarks = shot.get_valid_landmarks()
        for track in valid_landmarks:
            if track.id in reconstruction.points:
                points.add(track)

    candidate_shots = set(reconstruction.shots) - set(shot_ids)
    common_points = defaultdict(int)
    for track in points:
        neighbors = track.get_observations()
        for neighbor in neighbors:
            if neighbor.id in candidate_shots:
                common_points[neighbor] += 1

    pairs = sorted(common_points.items(), key=lambda x: -x[1])
    neighbors = set()
    for neighbor, num_points in pairs[:max_neighbors]:
        if num_points >= min_common_points:
            neighbors.add(neighbor.id)
        else:
            break
    return neighbors


def pairwise_reconstructability(common_tracks: int, rotation_inliers: int) -> float:
    """Likeliness of an image pair giving a good initial reconstruction."""
    outliers = common_tracks - rotation_inliers
    outlier_ratio = float(outliers) / common_tracks
    if outlier_ratio >= 0.3:
        return outliers
    else:
        return 0


TPairArguments = Tuple[
    str, str, NDArray, NDArray, pygeometry.Camera, pygeometry.Camera, float
]


def compute_image_pairs(
    track_dict: Dict[Tuple[str, str], tracking.TPairTracks], data: DataSetBase
) -> List[Tuple[str, str]]:
    """All matched image pairs sorted by reconstructability."""
    cameras = data.load_camera_models()
    args = _pair_reconstructability_arguments(track_dict, cameras, data)
    processes = data.config["processes"]
    result = parallel_map(_compute_pair_reconstructability, args, processes)
    result = list(result)
    pairs = [(im1, im2) for im1, im2, r in result if r > 0]
    score = [r for im1, im2, r in result if r > 0]
    order = np.argsort(-np.array(score))
    return [pairs[o] for o in order]


def _pair_reconstructability_arguments(
    track_dict: Dict[Tuple[str, str], tracking.TPairTracks],
    cameras: Dict[str, pygeometry.Camera],
    data: DataSetBase,
) -> List[TPairArguments]:
    threshold = 4 * data.config["five_point_algo_threshold"]
    args = []
    for (im1, im2), (_, p1, p2) in track_dict.items():
        camera1 = cameras[data.load_exif(im1)["camera"]]
        camera2 = cameras[data.load_exif(im2)["camera"]]
        args.append((im1, im2, p1, p2, camera1, camera2, threshold))
    return args


def _compute_pair_reconstructability(args: TPairArguments) -> Tuple[str, str, float]:
    log.setup()
    im1, im2, p1, p2, camera1, camera2, threshold = args
    R, inliers = two_view_reconstruction_rotation_only(
        p1, p2, camera1, camera2, threshold
    )
    r = pairwise_reconstructability(len(p1), len(inliers))
    return (im1, im2, r)


def add_shot(
    data: DataSetBase,
    reconstruction: types.Reconstruction,
    rig_assignments: Dict[str, Tuple[str, str, List[str]]],
    shot_id: str,
    pose: pygeometry.Pose,
) -> Set[str]:
    """Add a shot to the reconstruction.

    In case of a shot belonging to a rig instance, the pose of
    shot will drive the initial pose setup of the rig instance.
    All necessary shots and rig models will be created.
    """

    added_shots = set()
    if shot_id not in rig_assignments:
        camera_id = data.load_exif(shot_id)["camera"]
        shot = reconstruction.create_shot(shot_id, camera_id, pose)
        shot.metadata = helpers.get_image_metadata(data, shot_id)
        added_shots = {shot_id}
    else:
        instance_id, _, instance_shots = rig_assignments[shot_id]
        rig_instance = reconstruction.add_rig_instance(pymap.RigInstance(instance_id))

        for shot in instance_shots:
            _, rig_camera_id, _ = rig_assignments[shot]
            camera_id = data.load_exif(shot)["camera"]
            created_shot = reconstruction.create_shot(
                shot,
                camera_id,
                pygeometry.Pose(),
                rig_camera_id,
                instance_id,
            )
            created_shot.metadata = helpers.get_image_metadata(data, shot)
        rig_instance.update_instance_pose_with_shot(shot_id, pose)
        added_shots = set(instance_shots)

    return added_shots


def _two_view_reconstruction_inliers(
    b1: NDArray, b2: NDArray, R: NDArray, t: NDArray, threshold: float
) -> List[int]:
    """Returns indices of matches that can be triangulated."""
    ok = matching.compute_inliers_bearings(b1, b2, R, t, threshold)
    # pyre-fixme[7]: Expected `List[int]` but got `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    return np.nonzero(ok)[0]


def two_view_reconstruction_plane_based(
    b1: NDArray,
    b2: NDArray,
    threshold: float,
) -> Tuple[Optional[NDArray], Optional[NDArray], List[int]]:
    """Reconstruct two views from point correspondences lying on a plane.

    Args:
        b1, b2: lists bearings in the images
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    x1 = multiview.euclidean(b1)
    x2 = multiview.euclidean(b2)

    H, inliers = cv2.findHomography(x1, x2, cv2.RANSAC, threshold)
    motions = multiview.motion_from_plane_homography(H)

    if not motions:
        return None, None, []

    if len(motions) == 0:
        return None, None, []

    motion_inliers = []
    for R, t, _, _ in motions:
        inliers = _two_view_reconstruction_inliers(b1, b2, R.T, -R.T.dot(t), threshold)
        motion_inliers.append(inliers)

    # pyre-fixme[6]: For 1st argument expected `Union[_SupportsArray[dtype[typing.Any...
    best = np.argmax(map(len, motion_inliers))
    R, t, n, d = motions[best]
    inliers = motion_inliers[best]
    return cv2.Rodrigues(R)[0].ravel(), t, inliers


def two_view_reconstruction_and_refinement(
    b1: NDArray,
    b2: NDArray,
    R: NDArray,
    t: NDArray,
    threshold: float,
    iterations: int,
    transposed: bool,
) -> Tuple[NDArray, NDArray, List[int]]:
    """Reconstruct two views using provided rotation and translation.

    Args:
        b1, b2: lists bearings in the images
        R, t: rotation & translation
        threshold: reprojection error threshold
        iterations: number of iteration for refinement
        transposed: use transposed R, t instead

    Returns:
        rotation, translation and inlier list
    """
    if transposed:
        t_curr = -R.T.dot(t)
        R_curr = R.T
    else:
        t_curr = t.copy()
        R_curr = R.copy()

    inliers = _two_view_reconstruction_inliers(b1, b2, R_curr, t_curr, threshold)

    if len(inliers) > 5:
        T = multiview.relative_pose_optimize_nonlinear(
            b1[inliers], b2[inliers], t_curr, R_curr, iterations
        )
        R_curr = T[:, :3]
        t_curr = T[:, 3]
        inliers = _two_view_reconstruction_inliers(b1, b2, R_curr, t_curr, threshold)

    return cv2.Rodrigues(R_curr.T)[0].ravel(), -R_curr.T.dot(t_curr), inliers


def _two_view_rotation_inliers(
    b1: NDArray, b2: NDArray, R: NDArray, threshold: float
) -> List[int]:
    br2 = R.dot(b2.T).T
    ok = np.linalg.norm(br2 - b1, axis=1) < threshold
    # pyre-fixme[7]: Expected `List[int]` but got `ndarray[typing.Any,
    #  dtype[typing.Any]]`.
    return np.nonzero(ok)[0]


def two_view_reconstruction_rotation_only(
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    threshold: float,
) -> Tuple[NDArray, List[int]]:
    """Find rotation between two views from point correspondences.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    R = multiview.relative_pose_ransac_rotation_only(b1, b2, threshold, 1000, 0.999)
    inliers = _two_view_rotation_inliers(b1, b2, R, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), inliers


def two_view_reconstruction_5pt(
    b1: NDArray,
    b2: NDArray,
    R: NDArray,
    t: NDArray,
    threshold: float,
    iterations: int,
    check_reversal: bool = False,
    reversal_ratio: float = 1.0,
) -> Tuple[Optional[NDArray], Optional[NDArray], List[int]]:
    """Run 5-point reconstruction and refinement, given computed relative rotation and translation.

    Optionally, the method will perform reconstruction and refinement for both given and transposed
    rotation and translation.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold
        iterations: number of step for the non-linear refinement of the relative pose
        check_reversal: whether to check for Necker reversal ambiguity
        reversal_ratio: ratio of triangulated point between normal and reversed
                        configuration to consider a pair as being ambiguous

    Returns:
        rotation, translation and inlier list
    """

    configurations = [False, True] if check_reversal else [False]

    # Refine both normal and transposed relative motion
    results_5pt = []
    for transposed in configurations:
        R_5p, t_5p, inliers_5p = two_view_reconstruction_and_refinement(
            b1,
            b2,
            R,
            t,
            threshold,
            iterations,
            transposed,
        )

        valid_curr_5pt = R_5p is not None and t_5p is not None
        if len(inliers_5p) <= 5 or not valid_curr_5pt:
            continue

        logger.info(
            f"Two-view 5-points reconstruction inliers (transposed={transposed}): {len(inliers_5p)} / {len(b1)}"
        )
        results_5pt.append((R_5p, t_5p, inliers_5p))

    # Use relative motion if one version stands out
    if len(results_5pt) == 1:
        R_5p, t_5p, inliers_5p = results_5pt[0]
    elif len(results_5pt) == 2:
        inliers1, inliers2 = results_5pt[0][2], results_5pt[1][2]
        len1, len2 = len(inliers1), len(inliers2)
        ratio = min(len1, len2) / max(len1, len2)
        if ratio > reversal_ratio:
            logger.warning(
                f"Un-decidable Necker configuration (ratio={ratio}), skipping."
            )
            R_5p, t_5p, inliers_5p = None, None, []
        else:
            index = 0 if len1 > len2 else 1
            R_5p, t_5p, inliers_5p = results_5pt[index]
    else:
        R_5p, t_5p, inliers_5p = None, None, []

    return R_5p, t_5p, inliers_5p


def two_view_reconstruction_general(  # pyre-ignore[3]: pyre is not happy with the Dict[str, Any]
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    threshold: float,
    iterations: int,
    check_reversal: bool = False,
    reversal_ratio: float = 1.0,
) -> Tuple[Optional[NDArray], Optional[NDArray], List[int], Dict[str, Any]]:
    """Reconstruct two views from point correspondences.

    These will try different reconstruction methods and return the
    results of the one with most inliers.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold
        iterations: number of step for the non-linear refinement of the relative pose
        check_reversal: whether to check for Necker reversal ambiguity
        reversal_ratio: ratio of triangulated point between normal and reversed
                        configuration to consider a pair as being ambiguous

    Returns:
        rotation, translation and inlier list
    """

    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    # Get 5-point relative motion
    T_robust = multiview.relative_pose_ransac(b1, b2, threshold, 1000, 0.999)
    R_robust = T_robust[:, :3]
    t_robust = T_robust[:, 3]
    R_5p, t_5p, inliers_5p = two_view_reconstruction_5pt(
        b1,
        b2,
        R_robust,
        t_robust,
        threshold,
        iterations,
        check_reversal,
        reversal_ratio,
    )
    valid_5pt = R_5p is not None and t_5p is not None

    # Compute plane-based relative-motion
    R_plane, t_plane, inliers_plane = two_view_reconstruction_plane_based(
        b1,
        b2,
        threshold,
    )
    valid_plane = R_plane is not None and t_plane is not None

    report: Dict[str, Any] = {
        "5_point_inliers": len(inliers_5p),
        "plane_based_inliers": len(inliers_plane),
    }

    if valid_5pt and len(inliers_5p) > len(inliers_plane):
        report["method"] = "5_point"
        R, t, inliers = R_5p, t_5p, inliers_5p
    elif valid_plane:
        report["method"] = "plane_based"
        R, t, inliers = R_plane, t_plane, inliers_plane
    else:
        report["decision"] = "Could not find initial motion"
        logger.info(report["decision"])
        R, t, inliers = None, None, []
    return R, t, inliers, report


def reconstruction_from_relative_pose(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
    R: NDArray,
    t: NDArray,
) -> Tuple[Optional[types.Reconstruction], Dict[str, Any]]:
    """Create a reconstruction from 'im1' and 'im2' using the provided rotation 'R' and translation 't'."""
    report = {}

    min_inliers = data.config["five_point_algo_min_inliers"]

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()
    rig_assignments = rig.rig_assignments_per_image(data.load_rig_assignments())

    reconstruction = types.Reconstruction()
    reconstruction.reference = data.load_reference()
    reconstruction.cameras = camera_priors
    reconstruction.rig_cameras = rig_camera_priors

    new_shots = add_shot(data, reconstruction, rig_assignments, im1, pygeometry.Pose())

    if im2 not in new_shots:
        new_shots |= add_shot(
            data, reconstruction, rig_assignments, im2, pygeometry.Pose(R, t)
        )

    align_reconstruction(reconstruction, [], data.config)
    triangulate_shot_features(tracks_manager, reconstruction, new_shots, data.config)

    logger.info("Triangulated: {}".format(len(reconstruction.points)))
    report["triangulated_points"] = len(reconstruction.points)
    if len(reconstruction.points) < min_inliers:
        report["decision"] = "Initial motion did not generate enough points"
        logger.info(report["decision"])
        return None, report

    to_adjust = {s for s in new_shots if s != im1}
    bundle_shot_poses(
        reconstruction, to_adjust, camera_priors, rig_camera_priors, data.config
    )

    retriangulate(tracks_manager, reconstruction, data.config)
    if len(reconstruction.points) < min_inliers:
        report["decision"] = (
            "Re-triangulation after initial motion did not generate enough points"
        )
        logger.info(report["decision"])
        return None, report

    bundle_shot_poses(
        reconstruction, to_adjust, camera_priors, rig_camera_priors, data.config
    )

    report["decision"] = "Success"
    report["memory_usage"] = current_memory_usage()
    return reconstruction, report


def bootstrap_reconstruction(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
    p1: NDArray,
    p2: NDArray,
) -> Tuple[Optional[types.Reconstruction], Dict[str, Any]]:
    """Start a reconstruction using two shots."""
    logger.info("Starting reconstruction with {} and {}".format(im1, im2))
    report: Dict[str, Any] = {
        "image_pair": (im1, im2),
        "common_tracks": len(p1),
    }

    camera_priors = data.load_camera_models()
    camera1 = camera_priors[data.load_exif(im1)["camera"]]
    camera2 = camera_priors[data.load_exif(im2)["camera"]]

    threshold = data.config["five_point_algo_threshold"]
    iterations = data.config["five_point_refine_rec_iterations"]
    check_reversal = data.config["five_point_reversal_check"]
    reversal_ratio = data.config["five_point_reversal_ratio"]

    (
        R,
        t,
        inliers,
        report["two_view_reconstruction"],
    ) = two_view_reconstruction_general(
        p1, p2, camera1, camera2, threshold, iterations, check_reversal, reversal_ratio
    )

    if R is None or t is None:
        return None, report

    rec, rec_report = reconstruction_from_relative_pose(
        data, tracks_manager, im1, im2, R, t
    )
    report.update(rec_report)

    return rec, report


def reconstructed_points_for_images(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    images: Set[str],
) -> List[Tuple[str, int]]:
    """Number of reconstructed points visible on each image.

    Returns:
        A list of (image, num_point) pairs sorted by decreasing number
        of points.
    """
    non_reconstructed = [im for im in images if im not in reconstruction.shots]
    res = pysfm.count_tracks_per_shot(
        tracks_manager, non_reconstructed, list(reconstruction.points.keys())
    )
    return sorted(res.items(), key=lambda x: -x[1])


def resect(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    shot_id: str,
    threshold: float,
    min_inliers: int,
) -> Tuple[bool, Set[str], Dict[str, Any]]:
    """Try resecting and adding a shot to the reconstruction.

    Return:
        True on success.
    """

    rig_assignments = rig.rig_assignments_per_image(data.load_rig_assignments())
    camera = reconstruction.cameras[data.load_exif(shot_id)["camera"]]

    bs, Xs, ids = [], [], []
    for track, obs in tracks_manager.get_shot_observations(shot_id).items():
        if track in reconstruction.points:
            b = camera.pixel_bearing(obs.point)
            bs.append(b)
            Xs.append(reconstruction.points[track].coordinates)
            ids.append(track)
    bs = np.array(bs)
    Xs = np.array(Xs)
    if len(bs) < 5:
        return False, set(), {"num_common_points": len(bs)}

    T = multiview.absolute_pose_ransac(bs, Xs, threshold, 1000, 0.999)

    R = T[:, :3]
    t = T[:, 3]

    reprojected_bs = R.T.dot((Xs - t).T).T
    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

    inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
    ninliers = int(sum(inliers))

    logger.info("{} resection inliers: {} / {}".format(shot_id, ninliers, len(bs)))
    report: Dict[str, Any] = {
        "num_common_points": len(bs),
        "num_inliers": ninliers,
    }
    if ninliers >= min_inliers:
        R = T[:, :3].T
        t = -R.dot(T[:, 3])
        assert shot_id not in reconstruction.shots

        new_shots = add_shot(
            data, reconstruction, rig_assignments, shot_id, pygeometry.Pose(R, t)
        )

        if shot_id in rig_assignments:
            triangulate_shot_features(
                tracks_manager, reconstruction, new_shots, data.config
            )
        for i, succeed in enumerate(inliers):
            if succeed:
                add_observation_to_reconstruction(
                    tracks_manager, reconstruction, shot_id, ids[i]
                )
        report["shots"] = list(new_shots)
        return True, new_shots, report
    else:
        return False, set(), report


def corresponding_tracks(
    tracks1: Dict[str, pymap.Observation], tracks2: Dict[str, pymap.Observation]
) -> List[Tuple[str, str]]:
    features1 = {obs.id: t1 for t1, obs in tracks1.items()}
    corresponding_tracks = []
    for t2, obs in tracks2.items():
        feature_id = obs.id
        if feature_id in features1:
            corresponding_tracks.append((features1[feature_id], t2))
    return corresponding_tracks


def compute_common_tracks(
    reconstruction1: types.Reconstruction,
    reconstruction2: types.Reconstruction,
    tracks_manager1: pymap.TracksManager,
    tracks_manager2: pymap.TracksManager,
) -> List[Tuple[str, str]]:
    common_tracks = set()
    common_images = set(reconstruction1.shots.keys()).intersection(
        reconstruction2.shots.keys()
    )

    all_shot_ids1 = set(tracks_manager1.get_shot_ids())
    all_shot_ids2 = set(tracks_manager2.get_shot_ids())
    for image in common_images:
        if image not in all_shot_ids1 or image not in all_shot_ids2:
            continue
        at_shot1 = tracks_manager1.get_shot_observations(image)
        at_shot2 = tracks_manager2.get_shot_observations(image)
        for t1, t2 in corresponding_tracks(at_shot1, at_shot2):
            if t1 in reconstruction1.points and t2 in reconstruction2.points:
                common_tracks.add((t1, t2))
    return list(common_tracks)


def resect_reconstruction(
    reconstruction1: types.Reconstruction,
    reconstruction2: types.Reconstruction,
    tracks_manager1: pymap.TracksManager,
    tracks_manager2: pymap.TracksManager,
    threshold: float,
    min_inliers: int,
) -> Tuple[bool, NDArray, List[Tuple[str, str]]]:
    """Compute a similarity transform `similarity` such as :

    reconstruction2 = T . reconstruction1

    between two reconstruction 'reconstruction1' and 'reconstruction2'.

    Their respective tracks managers are used to find common tracks that
    are further used to compute the 3D similarity transform T using RANSAC.
    """

    common_tracks = compute_common_tracks(
        reconstruction1, reconstruction2, tracks_manager1, tracks_manager2
    )
    worked, similarity, inliers = align_two_reconstruction(
        reconstruction1, reconstruction2, common_tracks, threshold
    )
    if not worked or similarity is None:
        return False, np.ones((4, 4)), []

    inliers = [common_tracks[inliers[i]] for i in range(len(inliers))]
    return True, similarity, inliers


def add_observation_to_reconstruction(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    shot_id: str,
    track_id: str,
) -> None:
    observation = tracks_manager.get_observation(shot_id, track_id)
    reconstruction.add_observation(shot_id, track_id, observation)


class TrackHandlerBase(ABC):
    """Interface for providing/retrieving tracks from/to 'TrackTriangulator'."""

    @abstractmethod
    def get_observations(self, track_id: str) -> Dict[str, pymap.Observation]:
        """Returns the observations of 'track_id'"""
        pass

    @abstractmethod
    def store_track_coordinates(self, track_id: str, coordinates: NDArray) -> None:
        """Stores coordinates of triangulated track."""
        pass

    @abstractmethod
    def store_inliers_observation(self, track_id: str, shot_id: str) -> None:
        """Called by the 'TrackTriangulator' for each track inlier found."""
        pass


class TrackHandlerTrackManager(TrackHandlerBase):
    """Provider that reads tracks from a 'TrackManager' object."""

    tracks_manager: pymap.TracksManager
    reconstruction: types.Reconstruction

    def __init__(
        self,
        tracks_manager: pymap.TracksManager,
        reconstruction: types.Reconstruction,
    ) -> None:
        self.tracks_manager = tracks_manager
        self.reconstruction = reconstruction

    def get_observations(self, track_id: str) -> Dict[str, pymap.Observation]:
        """Return the observations of 'track_id', for all
        shots that appears in 'self.reconstruction.shots'
        """
        return {
            k: v
            for k, v in self.tracks_manager.get_track_observations(track_id).items()
            if k in self.reconstruction.shots
        }

    def store_track_coordinates(self, track_id: str, coordinates: NDArray) -> None:
        """Stores coordinates of triangulated track."""
        self.reconstruction.create_point(track_id, coordinates)

    def store_inliers_observation(self, track_id: str, shot_id: str) -> None:
        """Stores triangulation inliers in the tracks manager."""
        observation = self.tracks_manager.get_observation(shot_id, track_id)
        self.reconstruction.add_observation(shot_id, track_id, observation)


class TrackTriangulator:
    """Triangulate tracks in a reconstruction.

    Caches shot origin and rotation matrix
    """

    # for getting shots
    reconstruction: types.Reconstruction

    # for storing tracks inliers
    tracks_handler: TrackHandlerBase

    # caches
    origins: Dict[str, NDArray] = {}
    rotation_inverses: Dict[str, NDArray] = {}
    Rts: Dict[str, NDArray] = {}

    def __init__(
        self, reconstruction: types.Reconstruction, tracks_handler: TrackHandlerBase
    ) -> None:
        """Build a triangulator for a specific reconstruction."""
        self.reconstruction = reconstruction
        self.tracks_handler = tracks_handler
        self.origins: Dict[str, NDArray] = {}
        self.rotation_inverses: Dict[str, NDArray] = {}
        self.Rts: Dict[str, NDArray] = {}

    def triangulate_robust(
        self,
        track: str,
        reproj_threshold: float,
        min_ray_angle_degrees: float,
        min_depth: float,
        iterations: int,
    ) -> None:
        """Triangulate track in a RANSAC way and add point to reconstruction."""
        os, bs, ids = [], [], []
        for shot_id, obs in self.tracks_handler.get_observations(track).items():
            shot = self.reconstruction.shots[shot_id]
            os.append(self._shot_origin(shot))
            b = shot.camera.pixel_bearing(np.array(obs.point))
            r = self._shot_rotation_inverse(shot)
            bs.append(r.dot(b))
            ids.append(shot_id)

        if len(ids) < 2:
            return

        os = np.array(os)
        bs = np.array(bs)

        best_inliers = []
        best_point = None
        combinatiom_tried = set()
        ransac_tries = 11  # 0.99 proba, 60% inliers
        all_combinations = list(combinations(range(len(ids)), 2))

        thresholds = len(os) * [reproj_threshold]
        min_ray_angle_radians = np.radians(min_ray_angle_degrees)
        for i in range(ransac_tries):
            random_id = int(np.random.rand() * (len(all_combinations) - 1))
            if random_id in combinatiom_tried:
                continue

            i, j = all_combinations[random_id]
            combinatiom_tried.add(random_id)

            os_t = np.array([os[i], os[j]])
            bs_t = np.array([bs[i], bs[j]])

            valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
                os_t,
                bs_t,
                thresholds,
                min_ray_angle_radians,
                min_depth,
            )
            X = pygeometry.point_refinement(os_t, bs_t, X, iterations)

            if valid_triangulation:
                reprojected_bs = X - os
                reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]
                inliers = np.nonzero(
                    np.linalg.norm(reprojected_bs - bs, axis=1) < reproj_threshold
                )[0].tolist()

                if len(inliers) > len(best_inliers):
                    _, new_X = pygeometry.triangulate_bearings_midpoint(
                        os[inliers],
                        bs[inliers],
                        len(inliers) * [reproj_threshold],
                        min_ray_angle_radians,
                        min_depth,
                    )
                    new_X = pygeometry.point_refinement(
                        os[inliers], bs[inliers], X, iterations
                    )

                    reprojected_bs = new_X - os
                    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[
                        :, np.newaxis
                    ]
                    ls_inliers = np.nonzero(
                        np.linalg.norm(reprojected_bs - bs, axis=1) < reproj_threshold
                    )[0]
                    if len(ls_inliers) > len(inliers):
                        best_inliers = ls_inliers
                        best_point = new_X.tolist()
                    else:
                        best_inliers = inliers
                        best_point = X.tolist()

                    pout = 0.99
                    inliers_ratio = float(len(best_inliers)) / len(ids)
                    if inliers_ratio == 1.0:
                        break
                    optimal_iter = math.log(1.0 - pout) / math.log(
                        1.0 - inliers_ratio * inliers_ratio
                    )
                    if optimal_iter <= i:
                        break

        if len(best_inliers) > 1:
            self.tracks_handler.store_track_coordinates(track, best_point)
            for i in best_inliers:
                self.tracks_handler.store_inliers_observation(track, ids[i])

    def triangulate(
        self,
        track: str,
        reproj_threshold: float,
        min_ray_angle_degrees: float,
        min_depth: float,
        iterations: int,
    ) -> None:
        """Triangulate track and add point to reconstruction."""
        os, bs, ids = [], [], []
        for shot_id, obs in self.tracks_handler.get_observations(track).items():
            shot = self.reconstruction.shots[shot_id]
            os.append(self._shot_origin(shot))
            b = shot.camera.pixel_bearing(np.array(obs.point))
            r = self._shot_rotation_inverse(shot)
            bs.append(r.dot(b))
            ids.append(shot_id)

        if len(os) >= 2:
            thresholds = len(os) * [reproj_threshold]
            min_ray_angle_radians = np.radians(min_ray_angle_degrees)
            valid_triangulation, X = pygeometry.triangulate_bearings_midpoint(
                np.asarray(os),
                np.asarray(bs),
                thresholds,
                min_ray_angle_radians,
                min_depth,
            )
            if valid_triangulation:
                X = pygeometry.point_refinement(
                    np.array(os), np.array(bs), X, iterations
                )
                self.tracks_handler.store_track_coordinates(track, X.tolist())
                for shot_id in ids:
                    self.tracks_handler.store_inliers_observation(track, shot_id)

    def triangulate_dlt(
        self,
        track: str,
        reproj_threshold: float,
        min_ray_angle_degrees: float,
        min_depth: float,
        iterations: int,
    ) -> None:
        """Triangulate track using DLT and add point to reconstruction."""
        Rts, bs, os, ids = [], [], [], []
        for shot_id, obs in self.tracks_handler.get_observations(track).items():
            shot = self.reconstruction.shots[shot_id]
            os.append(self._shot_origin(shot))
            Rts.append(self._shot_Rt(shot))
            b = shot.camera.pixel_bearing(np.array(obs.point))
            bs.append(b)
            ids.append(shot_id)

        if len(Rts) >= 2:
            e, X = pygeometry.triangulate_bearings_dlt(
                # pyre-fixme[6]: For 1st argument expected `List[ndarray[typing.Any,
                #  typing.Any]]` but got `ndarray[typing.Any, dtype[typing.Any]]`.
                np.asarray(Rts),
                np.asarray(bs),
                reproj_threshold,
                np.radians(min_ray_angle_degrees),
                min_depth,
            )
            if e:
                X = pygeometry.point_refinement(
                    np.array(os), np.array(bs), X, iterations
                )
                self.tracks_handler.store_track_coordinates(track, X.tolist())
                for shot_id in ids:
                    self.tracks_handler.store_inliers_observation(track, shot_id)

    def _shot_origin(self, shot: pymap.Shot) -> NDArray:
        if shot.id in self.origins:
            return self.origins[shot.id]
        else:
            o = shot.pose.get_origin()
            self.origins[shot.id] = o
            return o

    def _shot_rotation_inverse(self, shot: pymap.Shot) -> NDArray:
        if shot.id in self.rotation_inverses:
            return self.rotation_inverses[shot.id]
        else:
            r = shot.pose.get_rotation_matrix().T
            self.rotation_inverses[shot.id] = r
            return r

    def _shot_Rt(self, shot: pymap.Shot) -> NDArray:
        if shot.id in self.Rts:
            return self.Rts[shot.id]
        else:
            r = shot.pose.get_Rt()
            self.Rts[shot.id] = r
            return r


def triangulate_shot_features(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    shot_ids: Set[str],
    config: Dict[str, Any],
) -> None:
    """Reconstruct as many tracks seen in shot_id as possible."""
    reproj_threshold = config["triangulation_threshold"]
    min_ray_angle = config["triangulation_min_ray_angle"]
    min_depth = config["triangulation_min_depth"]
    refinement_iterations = config["triangulation_refinement_iterations"]

    triangulator = TrackTriangulator(
        reconstruction, TrackHandlerTrackManager(tracks_manager, reconstruction)
    )

    all_shots_ids = set(tracks_manager.get_shot_ids())
    tracks_ids = {
        t
        for s in shot_ids
        if s in all_shots_ids
        for t in tracks_manager.get_shot_observations(s)
    }
    for track in tracks_ids:
        if track not in reconstruction.points:
            if config["triangulation_type"] == "ROBUST":
                triangulator.triangulate_robust(
                    track,
                    reproj_threshold,
                    min_ray_angle,
                    min_depth,
                    refinement_iterations,
                )
            elif config["triangulation_type"] == "FULL":
                triangulator.triangulate(
                    track,
                    reproj_threshold,
                    min_ray_angle,
                    min_depth,
                    refinement_iterations,
                )


def retriangulate(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Retrianguate all points"""
    chrono = Chronometer()
    report = {}
    report["num_points_before"] = len(reconstruction.points)

    threshold = config["triangulation_threshold"]
    min_ray_angle = config["triangulation_min_ray_angle"]
    min_depth = config["triangulation_min_depth"]
    refinement_iterations = config["triangulation_refinement_iterations"]

    reconstruction.points = {}

    all_shots_ids = set(tracks_manager.get_shot_ids())

    triangulator = TrackTriangulator(
        reconstruction, TrackHandlerTrackManager(tracks_manager, reconstruction)
    )
    tracks = set()
    for image in reconstruction.shots.keys():
        if image in all_shots_ids:
            tracks.update(tracks_manager.get_shot_observations(image).keys())
    for track in tracks:
        if config["triangulation_type"] == "ROBUST":
            triangulator.triangulate_robust(
                track, threshold, min_ray_angle, min_depth, refinement_iterations
            )
        elif config["triangulation_type"] == "FULL":
            triangulator.triangulate(
                track, threshold, min_ray_angle, min_depth, refinement_iterations
            )

    report["num_points_after"] = len(reconstruction.points)
    chrono.lap("retriangulate")
    report["wall_time"] = chrono.total_time()
    return report


def get_error_distribution(points: Dict[str, pymap.Landmark]) -> Tuple[float, float]:
    all_errors = []
    for track in points.values():
        all_errors += track.reprojection_errors.values()
    robust_mean = np.median(all_errors, axis=0)
    robust_std = 1.486 * np.median(
        np.linalg.norm(np.array(all_errors) - robust_mean, axis=1)
    )
    return robust_mean, robust_std


def get_actual_threshold(
    config: Dict[str, Any], points: Dict[str, pymap.Landmark]
) -> float:
    filter_type = config["bundle_outlier_filtering_type"]
    if filter_type == "FIXED":
        return config["bundle_outlier_fixed_threshold"]
    elif filter_type == "AUTO":
        mean, std = get_error_distribution(points)
        return config["bundle_outlier_auto_ratio"] * np.linalg.norm(mean + std)
    else:
        return 1.0


def remove_outliers(
    reconstruction: types.Reconstruction,
    config: Dict[str, Any],
    points: Optional[Dict[str, pymap.Landmark]] = None,
) -> int:
    """Remove points with large reprojection error.

    A list of point ids to be processed can be given in ``points``.
    """
    if points is None:
        points = reconstruction.points
    threshold_sqr = get_actual_threshold(config, reconstruction.points) ** 2
    outliers = []
    for point_id in points:
        for shot_id, error in reconstruction.points[
            point_id
        ].reprojection_errors.items():
            error_sqr = error[0] ** 2 + error[1] ** 2
            if error_sqr > threshold_sqr:
                outliers.append((point_id, shot_id))

    track_ids = set()
    for track, shot_id in outliers:
        reconstruction.map.remove_observation(shot_id, track)
        track_ids.add(track)

    for track in track_ids:
        if track in reconstruction.points:
            lm = reconstruction.points[track]
            if lm.number_of_observations() < 2:
                reconstruction.map.remove_landmark(lm)

    logger.info("Removed outliers: {}".format(len(outliers)))
    return len(outliers)


def shot_lla_and_compass(
    shot: pymap.Shot, reference: types.TopocentricConverter
) -> Tuple[float, float, float, float]:
    """Lat, lon, alt and compass of the reconstructed shot position."""
    topo = shot.pose.get_origin()
    lat, lon, alt = reference.to_lla(*topo)

    dz = shot.pose.get_R_cam_to_world()[:, 2]
    angle = np.rad2deg(np.arctan2(dz[0], dz[1]))
    angle = (angle + 360) % 360
    return lat, lon, alt, angle


def align_two_reconstruction(
    r1: types.Reconstruction,
    r2: types.Reconstruction,
    common_tracks: List[Tuple[str, str]],
    threshold: float,
) -> Tuple[bool, Optional[NDArray], List[int]]:
    """Estimate similarity transform T between two,
    reconstructions r1 and r2 such as r2 = T . r1
    """
    t1, t2 = r1.points, r2.points

    if len(common_tracks) > 6:
        p1 = np.array([t1[t[0]].coordinates for t in common_tracks])
        p2 = np.array([t2[t[1]].coordinates for t in common_tracks])

        # 3 samples / 100 trials / 50% outliers = 0.99 probability
        # with probability = 1-(1-(1-outlier)^model)^trial
        T, inliers = multiview.fit_similarity_transform(
            p1, p2, max_iterations=100, threshold=threshold
        )
        if len(inliers) > 0:
            return True, T, list(inliers)
    return False, None, []


def merge_two_reconstructions(
    r1: types.Reconstruction,
    r2: types.Reconstruction,
    config: Dict[str, Any],
    threshold: float = 1,
) -> List[types.Reconstruction]:
    """Merge two reconstructions with common tracks IDs."""
    common_tracks = list(set(r1.points) & set(r2.points))
    worked, T, inliers = align_two_reconstruction(r1, r2, common_tracks, threshold)

    if T and worked and len(inliers) >= 10:
        s, A, b = multiview.decompose_similarity_transform(T)
        r1p = r1
        apply_similarity(r1p, s, A, b)
        r = r2
        r.shots.update(r1p.shots)
        r.points.update(r1p.points)
        align_reconstruction(r, [], config)
        return [r]
    else:
        return [r1, r2]


def merge_reconstructions(
    reconstructions: List[types.Reconstruction], config: Dict[str, Any]
) -> List[types.Reconstruction]:
    """Greedily merge reconstructions with common tracks."""
    num_reconstruction = len(reconstructions)
    ids_reconstructions = np.arange(num_reconstruction)
    remaining_reconstruction = ids_reconstructions
    reconstructions_merged = []
    num_merge = 0

    for i, j in combinations(ids_reconstructions, 2):
        if (i in remaining_reconstruction) and (j in remaining_reconstruction):
            r = merge_two_reconstructions(
                reconstructions[i], reconstructions[j], config
            )
            if len(r) == 1:
                remaining_reconstruction = list(set(remaining_reconstruction) - {i, j})
                for k in remaining_reconstruction:
                    rr = merge_two_reconstructions(r[0], reconstructions[k], config)
                    if len(r) == 2:
                        break
                    else:
                        r = rr
                        remaining_reconstruction = list(
                            set(remaining_reconstruction) - {k}
                        )
                reconstructions_merged.append(r[0])
                num_merge += 1

    for k in remaining_reconstruction:
        reconstructions_merged.append(reconstructions[k])

    logger.info("Merged {0} reconstructions".format(num_merge))

    return reconstructions_merged


def paint_reconstruction(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> None:
    """Set the color of the points from the color of the tracks."""
    for k, point in reconstruction.points.items():
        point.color = list(
            map(
                float,
                next(
                    iter(tracks_manager.get_track_observations(str(k)).values())
                ).color,
            )
        )


class ShouldBundle:
    """Helper to keep track of when to run bundle."""

    interval: int
    new_points_ratio: float
    reconstruction: types.Reconstruction

    def __init__(self, data: DataSetBase, reconstruction: types.Reconstruction) -> None:
        self.interval = data.config["bundle_interval"]
        self.new_points_ratio = data.config["bundle_new_points_ratio"]
        self.reconstruction = reconstruction
        self.done()

    def should(self) -> bool:
        max_points = self.num_points_last * self.new_points_ratio
        max_shots = self.num_shots_last + self.interval
        return (
            len(self.reconstruction.points) >= max_points
            or len(self.reconstruction.shots) >= max_shots
        )

    def done(self) -> None:
        self.num_points_last = len(self.reconstruction.points)
        self.num_shots_last = len(self.reconstruction.shots)


class ShouldRetriangulate:
    """Helper to keep track of when to re-triangulate."""

    active: bool
    ratio: float
    reconstruction: types.Reconstruction

    def __init__(self, data: DataSetBase, reconstruction: types.Reconstruction) -> None:
        self.active = data.config["retriangulation"]
        self.ratio = data.config["retriangulation_ratio"]
        self.reconstruction = reconstruction
        self.done()

    def should(self) -> bool:
        max_points = self.num_points_last * self.ratio
        return self.active and len(self.reconstruction.points) > max_points

    def done(self) -> None:
        self.num_points_last = len(self.reconstruction.points)


def grow_reconstruction(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    images: Set[str],
    gcp: List[pymap.GroundControlPoint],
) -> Tuple[types.Reconstruction, Dict[str, Any]]:
    """Incrementally add shots to an initial reconstruction."""
    config = data.config
    report = {"steps": []}

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()

    paint_reconstruction(data, tracks_manager, reconstruction)
    align_reconstruction(reconstruction, gcp, config)

    bundle(reconstruction, camera_priors, rig_camera_priors, None, config)
    remove_outliers(reconstruction, config)
    paint_reconstruction(data, tracks_manager, reconstruction)

    should_bundle = ShouldBundle(data, reconstruction)
    should_retriangulate = ShouldRetriangulate(data, reconstruction)
    while True:
        if config["save_partial_reconstructions"]:
            paint_reconstruction(data, tracks_manager, reconstruction)
            data.save_reconstruction(
                [reconstruction],
                "reconstruction.{}.json".format(
                    datetime.datetime.now().isoformat().replace(":", "_")
                ),
            )

        candidates = reconstructed_points_for_images(
            tracks_manager, reconstruction, images
        )
        if not candidates:
            break

        logger.info("-------------------------------------------------------")
        threshold = data.config["resection_threshold"]
        min_inliers = data.config["resection_min_inliers"]
        for image, _ in candidates:
            ok, new_shots, resrep = resect(
                data,
                tracks_manager,
                reconstruction,
                image,
                threshold,
                min_inliers,
            )
            if not ok:
                continue

            images -= new_shots
            bundle_shot_poses(
                reconstruction,
                new_shots,
                camera_priors,
                rig_camera_priors,
                data.config,
            )

            logger.info(f"Adding {' and '.join(new_shots)} to the reconstruction")
            step: Dict[str, Union[List[int], List[str], int, List[int], Any]] = {
                "images": list(new_shots),
                "resection": resrep,
                "memory_usage": current_memory_usage(),
            }
            report["steps"].append(step)

            np_before = len(reconstruction.points)
            triangulate_shot_features(tracks_manager, reconstruction, new_shots, config)
            np_after = len(reconstruction.points)
            step["triangulated_points"] = np_after - np_before

            if should_retriangulate.should():
                logger.info("Re-triangulating")
                align_reconstruction(reconstruction, gcp, config)
                b1rep = bundle(
                    reconstruction, camera_priors, rig_camera_priors, None, config
                )
                rrep = retriangulate(tracks_manager, reconstruction, config)
                b2rep = bundle(
                    reconstruction, camera_priors, rig_camera_priors, None, config
                )
                remove_outliers(reconstruction, config)
                step["bundle"] = b1rep
                step["retriangulation"] = rrep
                step["bundle_after_retriangulation"] = b2rep
                should_retriangulate.done()
                should_bundle.done()
            elif should_bundle.should():
                align_reconstruction(reconstruction, gcp, config)
                brep = bundle(
                    reconstruction, camera_priors, rig_camera_priors, None, config
                )
                remove_outliers(reconstruction, config)
                step["bundle"] = brep
                should_bundle.done()
            elif config["local_bundle_radius"] > 0:
                bundled_points, brep = bundle_local(
                    reconstruction,
                    camera_priors,
                    rig_camera_priors,
                    None,
                    image,
                    config,
                )
                remove_outliers(reconstruction, config, bundled_points)
                step["local_bundle"] = brep

            logger.info(f"Reconstruction now has {len(reconstruction.shots)} shots.")

            break
        else:
            logger.info("Some images can not be added")
            break

    logger.info("-------------------------------------------------------")

    align_result = align_reconstruction(reconstruction, gcp, config, bias_override=True)
    if not align_result and config["bundle_compensate_gps_bias"]:
        overidden_config = config.copy()
        overidden_config["bundle_compensate_gps_bias"] = False
        config = overidden_config

    bundle(reconstruction, camera_priors, rig_camera_priors, gcp, config)
    remove_outliers(reconstruction, config)
    paint_reconstruction(data, tracks_manager, reconstruction)
    return reconstruction, report


def triangulation_reconstruction(
    data: DataSetBase, tracks_manager: pymap.TracksManager
) -> Tuple[Dict[str, Any], List[types.Reconstruction]]:
    """Run the triangulation reconstruction pipeline."""
    logger.info("Starting triangulation reconstruction")
    report = {}
    chrono = Chronometer()

    images = tracks_manager.get_shot_ids()
    data.init_reference(images)

    camera_priors = data.load_camera_models()
    rig_camera_priors = data.load_rig_cameras()
    gcp = data.load_ground_control_points()

    reconstruction = helpers.reconstruction_from_metadata(data, images)

    config = data.config
    config_override = config.copy()
    config_override["triangulation_type"] = "ROBUST"
    config_override["bundle_max_iterations"] = 10

    report["steps"] = []
    outer_iterations = 3
    inner_iterations = 5
    for i in range(outer_iterations):
        rrep = retriangulate(tracks_manager, reconstruction, config_override)
        triangulated_points = rrep["num_points_after"]
        logger.info(
            f"Triangulation SfM. Outer iteration {i}, triangulated {triangulated_points} points."
        )

        for j in range(inner_iterations):
            if config_override["save_partial_reconstructions"]:
                paint_reconstruction(data, tracks_manager, reconstruction)
                data.save_reconstruction(
                    [reconstruction], f"reconstruction.{i*inner_iterations+j}.json"
                )

            step = {}
            logger.info(f"Triangulation SfM. Inner iteration {j}, running bundle ...")
            align_reconstruction(reconstruction, gcp, config_override)
            b1rep = bundle(
                reconstruction, camera_priors, rig_camera_priors, None, config_override
            )
            remove_outliers(reconstruction, config_override)
            step["bundle"] = b1rep
            step["retriangulation"] = rrep
            report["steps"].append(step)

    logger.info("Triangulation SfM done.")
    logger.info("-------------------------------------------------------")
    chrono.lap("compute_reconstructions")
    report["wall_times"] = dict(chrono.lap_times())

    align_result = align_reconstruction(reconstruction, gcp, config, bias_override=True)
    if not align_result and config["bundle_compensate_gps_bias"]:
        overidden_bias_config = config.copy()
        overidden_bias_config["bundle_compensate_gps_bias"] = False
        config = overidden_bias_config

    bundle(reconstruction, camera_priors, rig_camera_priors, gcp, config)
    remove_outliers(reconstruction, config_override)
    paint_reconstruction(data, tracks_manager, reconstruction)
    return report, [reconstruction]


def _get_common_feature_arrays(
    tracks_manager: pymap.TracksManager,
    im1: str,
    im2: str,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return the feature arrays of common tracks between two images."""
    features1 = []
    features2 = []
    for _, p1, p2 in tracks_manager.get_all_common_observations(im1, im2):
        features1.append(p1.point)
        features2.append(p2.point)
    features1 = np.array(features1)
    features2 = np.array(features2)
    return (features1, features2)


def compute_image_pairs_sequential(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> list[tuple[str, str]]:
    """Compute all possible pairs of images that share at least `min_common` points.
    The pairs are sorted by "reconstructability" which is estimated by the
    pairwise_reconstructability function.
    """
    cameras = data.load_camera_models()
    threshold = 4 * data.config["five_point_algo_threshold"]
    results = []
    connectivity = tracks_manager.get_all_pairs_connectivity()
    for (im1, im2), size in connectivity.items():
        if size < min_common:
            continue
        features1, features2 = _get_common_feature_arrays(tracks_manager, im1, im2)
        camera1 = cameras[data.load_exif(im1)["camera"]]
        camera2 = cameras[data.load_exif(im2)["camera"]]
        result = _compute_pair_reconstructability(
            (im1, im2, features1, features2, camera1, camera2, threshold)
        )
        if result[2] > 0:
            results.append(result)
    results.sort(key=lambda x: x[2], reverse=True)
    return [(im1, im2) for im1, im2, _ in results]


def incremental_reconstruction(
    data: DataSetBase, tracks_manager: pymap.TracksManager
) -> Tuple[Dict[str, Any], List[types.Reconstruction]]:
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction")
    report = {}
    chrono = Chronometer()

    images = tracks_manager.get_shot_ids()

    data.init_reference(images)

    remaining_images = set(images)
    gcp = data.load_ground_control_points()
    logger.info(f"Loaded {len(gcp)} ground control points.")

    common_tracks = None
    if data.config["processes"] > 1:
        # Get pairs in parallel, preload all features for all
        # pairs of image. Features of an image can be stored
        # repeatedly in multiple image pairs.
        # Pros: fast; Cons: high memory usage
        common_tracks = tracking.all_common_tracks_with_features(tracks_manager)
        pairs = compute_image_pairs(common_tracks, data)
    else:
        # Get pairs sequentially, load features lazily.
        # Pros: low memory usage; Cons: slow
        pairs = compute_image_pairs_sequential(data, tracks_manager)
    logging.info(f"Estimated reconstructability of {len(pairs)} image pairs.")

    reconstructions = []
    chrono.lap("compute_image_pairs")
    report["num_candidate_image_pairs"] = len(pairs)
    report["reconstructions"] = []
    for im1, im2 in pairs:
        if im1 in remaining_images and im2 in remaining_images:
            rec_report = {}
            report["reconstructions"].append(rec_report)
            if common_tracks is not None:
                # Save time if the common features are preloaded
                _, p1, p2 = common_tracks[im1, im2]
            else:
                # At the sacrifice of lazily reloading the features, we avoid
                # storing features for all view pairs in memory
                p1, p2 = _get_common_feature_arrays(tracks_manager, im1, im2)
            reconstruction, rec_report["bootstrap"] = bootstrap_reconstruction(
                data, tracks_manager, im1, im2, p1, p2
            )

            if reconstruction:
                remaining_images -= set(reconstruction.shots)
                reconstruction, rec_report["grow"] = grow_reconstruction(
                    data,
                    tracks_manager,
                    reconstruction,
                    remaining_images,
                    gcp,
                )
                reconstructions.append(reconstruction)
                reconstructions = sorted(reconstructions, key=lambda x: -len(x.shots))

    for k, r in enumerate(reconstructions):
        logger.info(
            "Reconstruction {}: {} images, {} points".format(
                k, len(r.shots), len(r.points)
            )
        )
    logger.info("{} partial reconstructions in total.".format(len(reconstructions)))
    chrono.lap("compute_reconstructions")
    report["wall_times"] = dict(chrono.lap_times())
    report["not_reconstructed_images"] = list(remaining_images)
    return report, reconstructions


def reconstruct_from_prior(
    data: DataSetBase,
    tracks_manager: pymap.TracksManager,
    rec_prior: types.Reconstruction,
) -> Tuple[Dict[str, Any], types.Reconstruction]:
    """Retriangulate a new reconstruction from the rec_prior"""
    reconstruction = types.Reconstruction()
    reconstruction.reference = rec_prior.reference
    report = {}
    rec_report = {}
    report["retriangulate"] = [rec_report]
    images = tracks_manager.get_shot_ids()

    # copy prior poses, cameras
    reconstruction.cameras = rec_prior.cameras
    for shot in rec_prior.shots.values():
        reconstruction.add_shot(shot)
    prior_images = set(rec_prior.shots)
    remaining_images = set(images) - prior_images

    rec_report["num_prior_images"] = len(prior_images)
    rec_report["num_remaining_images"] = len(remaining_images)

    # Start with the known poses
    triangulate_shot_features(tracks_manager, reconstruction, prior_images, data.config)
    paint_reconstruction(data, tracks_manager, reconstruction)
    report["not_reconstructed_images"] = list(remaining_images)
    return report, reconstruction


class Chronometer:
    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        t = timer()
        lap = ("start", 0, t)
        self.laps = [lap]
        self.laps_dict = {"start": lap}

    def lap(self, key: str) -> None:
        t = timer()
        dt = t - self.laps[-1][2]
        lap = (key, dt, t)
        self.laps.append(lap)
        self.laps_dict[key] = lap

    def lap_time(self, key: str) -> float:
        return self.laps_dict[key][1]

    def lap_times(self) -> List[Tuple[str, float]]:
        return [(k, dt) for k, dt, t in self.laps[1:]]

    def total_time(self) -> float:
        return self.laps[-1][2] - self.laps[0][2]
