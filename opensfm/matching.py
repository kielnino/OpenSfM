# pyre-strict
import logging
import os
from timeit import default_timer as timer
from typing import Any, Dict, Generator, List, Optional, Set, Sized, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from opensfm import (
    context,
    feature_loader,
    log,
    multiview,
    pairs_selection,
    pyfeatures,
    pygeometry,
)
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


def clear_cache() -> None:
    feature_loader.instance.clear_cache()


def match_images(
    data: DataSetBase,
    config_override: Dict[str, Any],
    ref_images: List[str],
    cand_images: List[str],
) -> Tuple[Dict[Tuple[str, str], List[Tuple[int, int]]], Dict[str, Any]]:
    """Perform pair matchings between two sets of images.

    It will do matching for each pair (i, j), i being in
    ref_images and j in cand_images, taking assumption that
    matching(i, j) == matching(j ,i). This does not hold for
    non-symmetric matching options like WORDS. Data will be
    stored in i matching only.
    """

    # Get EXIFs data
    all_images = list(set(ref_images + cand_images))
    exifs = {im: data.load_exif(im) for im in all_images}

    # Generate pairs for matching
    pairs, preport = pairs_selection.match_candidates_from_metadata(
        ref_images,
        cand_images,
        exifs,
        data,
        config_override,
    )

    # Match them !
    return (
        match_images_with_pairs(data, config_override, exifs, pairs),
        preport,
    )


def match_images_with_pairs(
    data: DataSetBase,
    config_override: Dict[str, Any],
    exifs: Dict[str, Any],
    pairs: List[Tuple[str, str]],
    poses: Optional[Dict[str, pygeometry.Pose]] = None,
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """Perform pair matchings given pairs."""
    cameras = data.load_camera_models()
    args = list(match_arguments(pairs, data, config_override, cameras, exifs, poses))

    # Perform all pair matchings in parallel
    start = timer()
    logger.info("Matching {} image pairs".format(len(pairs)))
    processes = config_override.get("processes", data.config["processes"])
    mem_per_process = 512
    jobs_per_process = 2
    processes = context.processes_that_fit_in_memory(processes, mem_per_process)
    logger.info("Computing pair matching with %d processes" % processes)
    matches = context.parallel_map(match_unwrap_args, args, processes, jobs_per_process)
    logger.info(
        "Matched {} pairs {} in {} seconds ({} seconds/pair).".format(
            len(pairs),
            log_projection_types(pairs, exifs, cameras),
            timer() - start,
            (timer() - start) / len(pairs) if pairs else 0,
        )
    )

    # Index results per pair
    resulting_pairs = {}
    for im1, im2, m in matches:
        resulting_pairs[im1, im2] = m
    return resulting_pairs


def log_projection_types(
    pairs: List[Tuple[str, str]],
    exifs: Dict[str, Any],
    cameras: Dict[str, pygeometry.Camera],
) -> str:
    if not pairs:
        return ""

    projection_type_pairs = {}
    for im1, im2 in pairs:
        pt1 = cameras[exifs[im1]["camera"]].projection_type
        pt2 = cameras[exifs[im2]["camera"]].projection_type

        if pt1 not in projection_type_pairs:
            projection_type_pairs[pt1] = {}

        if pt2 not in projection_type_pairs[pt1]:
            projection_type_pairs[pt1][pt2] = []

        projection_type_pairs[pt1][pt2].append((im1, im2))

    output = "("
    for pt1 in projection_type_pairs:
        for pt2 in projection_type_pairs[pt1]:
            output += "{}-{}: {}, ".format(
                pt1, pt2, len(projection_type_pairs[pt1][pt2])
            )

    return output[:-2] + ")"


def save_matches(
    data: DataSetBase,
    images_ref: List[str],
    matched_pairs: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> None:
    """Given pairwise matches (image 1, image 2) - > matches,
    save them such as only {image E images_ref} will store the matches.
    """
    images_ref_set = set(images_ref)
    matches_per_im1 = {im: {} for im in images_ref}
    for (im1, im2), m in matched_pairs.items():
        if im1 in images_ref_set:
            matches_per_im1[im1][im2] = m
        elif im2 in images_ref_set:
            matches_per_im1[im2][im1] = m
        else:
            raise RuntimeError(
                "Couldn't save matches for {}. No image found in images_ref.".format(
                    (im1, im2)
                )
            )

    for im1, im1_matches in matches_per_im1.items():
        data.save_matches(im1, im1_matches)


def match_arguments(
    pairs: List[Tuple[str, str]],
    data: DataSetBase,
    config_override: Dict[str, Any],
    cameras: Dict[str, pygeometry.Camera],
    exifs: Dict[str, pygeometry.Camera],
    poses: Optional[Dict[str, pygeometry.Pose]],
) -> Generator[
    Tuple[
        str,
        str,
        Dict[str, pygeometry.Camera],
        Dict[str, pygeometry.Camera],
        DataSetBase,
        Dict[str, Any],
        Optional[Dict[str, pygeometry.Pose]],
    ],
    None,
    None,
]:
    """Generate arguments for parallel processing of pair matching"""
    for im1, im2 in pairs:
        yield im1, im2, cameras, exifs, data, config_override, poses


def match_unwrap_args(
    args: Tuple[
        str,
        str,
        Dict[str, pygeometry.Camera],
        Dict[str, Any],
        DataSetBase,
        Dict[str, Any],
        Optional[Dict[str, pygeometry.Pose]],
    ],
) -> Tuple[str, str, NDArray]:
    """Wrapper for parallel processing of pair matching.

    Compute all pair matchings of a given image and save them.
    """
    log.setup()
    im1 = args[0]
    im2 = args[1]
    cameras = args[2]
    exifs = args[3]
    data: DataSetBase = args[4]
    config_override = args[5]
    poses = args[6]
    if poses:
        pose1 = poses[im1]
        pose2 = poses[im2]
        pose = pose2.relative_to(pose1)
    else:
        pose = None
    camera1 = cameras[exifs[im1]["camera"]]
    camera2 = cameras[exifs[im2]["camera"]]
    matches = match(im1, im2, camera1, camera2, data, config_override, pose)
    return im1, im2, matches


def match_descriptors(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    config_override: Dict[str, Any],
) -> NDArray:
    """Perform descriptor matching for a pair of images."""
    # Override parameters
    overriden_config = data.config.copy()
    overriden_config.update(config_override)

    # Run descriptor matching
    time_start = timer()
    _, _, matches, matcher_type = _match_descriptors_impl(
        im1, im2, camera1, camera2, data, overriden_config
    )
    time_2d_matching = timer() - time_start

    # From indexes in filtered sets, to indexes in original sets of features
    matches_unfiltered = []
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None:
        matches_unfiltered = unfilter_matches(matches, m1, m2)

    symmetric = "symmetric" if overriden_config["symmetric_matching"] else "one-way"
    logger.debug(
        "Matching {} and {}.  Matcher: {} ({}) T-desc: {:1.3f} Matches: {}".format(
            im1,
            im2,
            matcher_type,
            symmetric,
            time_2d_matching,
            len(matches_unfiltered),
        )
    )
    return np.array(matches_unfiltered, dtype=int)


def _match_descriptors_guided_impl(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    relative_pose: pygeometry.Pose,
    data: DataSetBase,
    overriden_config: Dict[str, Any],
) -> Tuple[NDArray, NDArray, NDArray, str]:
    """Perform descriptor guided matching for a pair of images, using their relative pose. It also apply static objects removal."""
    guided_matcher_override = "BRUTEFORCE"
    matcher_type = overriden_config["matcher_type"].upper()
    symmetric_matching = overriden_config["symmetric_matching"]
    if matcher_type in ["WORDS", "FLANN"] or symmetric_matching:
        logger.warning(
            f"{matcher_type} and/or symmetric isn't supported for guided matching, switching to asymmetric {guided_matcher_override}"
        )
        matcher_type = guided_matcher_override

    # Will apply mask to features if any
    dummy = np.array([])
    segmentation_in_descriptor = overriden_config["matching_use_segmentation"]
    features_data1 = feature_loader.instance.load_all_data(
        data,
        im1,
        masked=True,
        segmentation_in_descriptor=segmentation_in_descriptor,
    )
    features_data2 = feature_loader.instance.load_all_data(
        data, im2, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    bearings1 = feature_loader.instance.load_bearings(
        data, im1, masked=True, camera=camera1
    )
    bearings2 = feature_loader.instance.load_bearings(
        data, im2, masked=True, camera=camera2
    )

    if (
        features_data1 is None
        or bearings1 is None
        or len(features_data1.points) < 2
        or features_data2 is None
        or bearings2 is None
        or len(features_data2.points) < 2
    ):
        return dummy, dummy, dummy, matcher_type

    d1 = features_data1.descriptors
    d2 = features_data2.descriptors
    if d1 is None or d2 is None:
        return dummy, dummy, dummy, matcher_type

    debug_label = (
        "guided"
        if overriden_config.get("debug_epipolar_images", False)
        else None
    )
    epipolar_mask = compute_inliers_bearing_epipolar(
        bearings1,
        bearings2,
        relative_pose,
        overriden_config["guided_matching_threshold"],
        debug_label=debug_label,
        camera1=camera1,
        camera2=camera2,
        p1=features_data1.points,
        p2=features_data2.points,
        data=data,
        im1=im1,
        im2=im2,
    )
    matches = match_brute_force_symmetric(d1, d2, overriden_config, epipolar_mask)

    # Adhoc filters
    if overriden_config["matching_use_filters"]:
        matches = apply_adhoc_filters(
            data,
            matches,
            im1,
            camera1,
            features_data1.points,
            im2,
            camera2,
            features_data2.points,
        )
    return (
        features_data1.points,
        features_data2.points,
        np.array(matches, dtype=int),
        matcher_type,
    )


def _match_descriptors_impl(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    overriden_config: Dict[str, Any],
) -> Tuple[NDArray, NDArray, NDArray, str]:
    """Perform descriptor matching for a pair of images. It also apply static objects removal."""
    dummy = np.array([])
    matcher_type = overriden_config["matcher_type"].upper()
    dummy_ret = dummy, dummy, dummy, matcher_type

    # Will apply mask to features if any
    dummy = np.array([])
    segmentation_in_descriptor = overriden_config["matching_use_segmentation"]
    features_data1 = feature_loader.instance.load_all_data(
        data, im1, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    features_data2 = feature_loader.instance.load_all_data(
        data, im2, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    if (
        features_data1 is None
        or len(features_data1.points) < 2
        or features_data2 is None
        or len(features_data2.points) < 2
    ):
        return dummy_ret

    d1 = features_data1.descriptors
    d2 = features_data2.descriptors
    if d1 is None or d2 is None:
        return dummy_ret

    symmetric_matching = overriden_config["symmetric_matching"]
    if matcher_type == "WORDS":
        words1 = feature_loader.instance.load_words(data, im1, masked=True)
        words2 = feature_loader.instance.load_words(data, im2, masked=True)
        if words1 is None or words2 is None:
            return dummy_ret

        if symmetric_matching:
            matches = match_words_symmetric(
                d1,
                words1,
                d2,
                words2,
                overriden_config,
            )
        else:
            matches = match_words(
                d1,
                words1,
                d2,
                words2,
                overriden_config,
            )

    elif matcher_type == "FLANN":
        f1 = feature_loader.instance.load_features_index(
            data,
            im1,
            masked=True,
            segmentation_in_descriptor=segmentation_in_descriptor,
        )
        if not f1:
            return dummy_ret
        feat_data_index1, index1 = f1
        if symmetric_matching:
            f2 = feature_loader.instance.load_features_index(
                data,
                im2,
                masked=True,
                segmentation_in_descriptor=segmentation_in_descriptor,
            )
            if not f2:
                return dummy_ret
            feat_data_index2, index2 = f2

            descriptors1 = feat_data_index1.descriptors
            descriptors2 = feat_data_index2.descriptors
            if descriptors1 is None or descriptors2 is None:
                return dummy_ret

            matches = match_flann_symmetric(
                descriptors1,
                index1,
                descriptors2,
                index2,
                overriden_config,
            )
        else:
            matches = match_flann(index1, d2, overriden_config)
    elif matcher_type == "BRUTEFORCE":
        if symmetric_matching:
            matches = match_brute_force_symmetric(d1, d2, overriden_config)
        else:
            matches = match_brute_force(d1, d2, overriden_config)
    else:
        raise ValueError("Invalid matcher_type: {}".format(matcher_type))

    # Adhoc filters
    if overriden_config["matching_use_filters"]:
        matches = apply_adhoc_filters(
            data,
            list(matches),
            im1,
            camera1,
            features_data1.points,
            im2,
            camera2,
            features_data2.points,
        )
    return (
        features_data1.points,
        features_data2.points,
        np.array(matches, dtype=int),
        matcher_type,
    )


def match_robust(
    im1: str,
    im2: str,
    matches: Sized,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    config_override: Dict[str, Any],
    input_is_masked: bool = True,
) -> NDArray:
    """Perform robust geometry matching on a set of matched descriptors indexes."""
    # Override parameters
    overriden_config = data.config.copy()
    overriden_config.update(config_override)

    # Will apply mask to features if any
    segmentation_in_descriptor = overriden_config[
        "matching_use_segmentation"
    ]  # unused but keep using the same cache
    features_data1 = feature_loader.instance.load_all_data(
        data,
        im1,
        masked=input_is_masked,
        segmentation_in_descriptor=segmentation_in_descriptor,
    )
    features_data2 = feature_loader.instance.load_all_data(
        data,
        im2,
        masked=input_is_masked,
        segmentation_in_descriptor=segmentation_in_descriptor,
    )
    if (
        features_data1 is None
        or len(features_data1.points) < 2
        or features_data2 is None
        or len(features_data2.points) < 2
    ):
        return np.array([])

    # Run robust matching
    np_matches = np.array(matches, dtype=int)
    t = timer()
    rmatches, _ = _match_robust_impl(
        im1,
        im2,
        features_data1.points,
        features_data2.points,
        np_matches,
        camera1,
        camera2,
        data,
        overriden_config,
    )
    time_robust_matching = timer() - t

    # From indexes in filtered sets, to indexes in original sets of features
    rmatches_unfiltered = []
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None and input_is_masked:
        rmatches_unfiltered = unfilter_matches(rmatches, m1, m2)
    else:
        rmatches_unfiltered = rmatches

    robust_matching_min_match = overriden_config["robust_matching_min_match"]
    logger.debug(
        "Matching {} and {}. T-robust: {:1.3f} "
        "Matches: {} Robust: {} Success: {}".format(
            im1,
            im2,
            time_robust_matching,
            len(matches),
            len(rmatches_unfiltered),
            len(rmatches_unfiltered) >= robust_matching_min_match,
        )
    )

    if len(rmatches_unfiltered) < robust_matching_min_match:
        return np.array([])
    return np.array(rmatches_unfiltered, dtype=int)


def _match_robust_impl(
    im1: str,
    im2: str,
    p1: NDArray,
    p2: NDArray,
    matches: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    overriden_config: Dict[str, Any],
) -> Tuple[NDArray, Optional[pygeometry.Pose]]:
    """Perform robust geometry matching on a set of matched descriptors indexes."""
    # robust matching
    rmatches, pose = robust_match(p1, p2, camera1, camera2, matches, overriden_config)
    rmatches = np.array([[a, b] for a, b in rmatches])
    return rmatches, pose


def match(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    data: DataSetBase,
    config_override: Dict[str, Any],
    guided_matching_pose: Optional[pygeometry.Pose],
) -> NDArray:
    """Perform full matching (descriptor+robust, optionally guided) for a pair of images."""
    # Override parameters
    overriden_config = data.config.copy()
    overriden_config.update(config_override)

    # Run descriptor matching
    time_start = timer()
    if guided_matching_pose:
        p1, p2, matches, matcher_type = _match_descriptors_guided_impl(
            im1, im2, camera1, camera2, guided_matching_pose, data, overriden_config
        )
    else:
        p1, p2, matches, matcher_type = _match_descriptors_impl(
            im1, im2, camera1, camera2, data, overriden_config
        )
    time_2d_matching = timer() - time_start

    symmetric = "symmetric" if overriden_config["symmetric_matching"] else "one-way"
    robust_matching_min_match = overriden_config["robust_matching_min_match"]
    if len(matches) < robust_matching_min_match:
        logger.debug(
            "Matching {} and {}.  Matcher: {} ({}) T-desc: {:1.3f} "
            "Matches: FAILED".format(
                im1, im2, matcher_type, symmetric, time_2d_matching
            )
        )
        return np.array([])

    # Run robust matching (non guided case only)
    t = timer()
    rmatches, pose = _match_robust_impl(
        im1, im2, p1, p2, matches, camera1, camera2, data, overriden_config
    )
    time_robust_matching = timer() - t

    # Run epipolar-guided post-matching to recover additional matches
    time_epipolar_guided = 0.0
    if (
        overriden_config["matching_epipolar_guided"]
        and pose is not None
        and len(rmatches) >= robust_matching_min_match
    ):
        t = timer()
        new_matches = _match_epipolar_guided_post_impl(
            im1, im2, camera1, camera2, p1, p2, rmatches, pose, data, overriden_config
        )
        time_epipolar_guided = timer() - t
        if len(new_matches) > 0:
            rmatches = np.vstack([rmatches, new_matches])

    # From indexes in filtered sets, to indexes in original sets of features
    m1 = feature_loader.instance.load_mask(data, im1)
    m2 = feature_loader.instance.load_mask(data, im2)
    if m1 is not None and m2 is not None:
        rmatches = unfilter_matches(rmatches, m1, m2)

    time_total = timer() - time_start

    logger.debug(
        "Matching {} and {}.  Matcher: {} ({}) "
        "T-desc: {:1.3f} T-robust: {:1.3f} T-epipolar: {:1.3f} T-total: {:1.3f} "
        "Matches: {} Robust: {} Success: {}".format(
            im1,
            im2,
            matcher_type,
            symmetric,
            time_2d_matching,
            time_robust_matching,
            time_epipolar_guided,
            time_total,
            len(matches),
            len(rmatches),
            len(rmatches) >= robust_matching_min_match,
        )
    )

    if len(rmatches) < robust_matching_min_match:
        return np.array([])
    return np.array(rmatches, dtype=int)


def _match_epipolar_guided_post_impl(
    im1: str,
    im2: str,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    p1: NDArray,
    p2: NDArray,
    rmatches: NDArray,
    pose: pygeometry.Pose,
    data: DataSetBase,
    overriden_config: Dict[str, Any],
) -> NDArray:
    """Find additional matches using epipolar geometry from a known relative pose.

    After robust matching has estimated a relative pose, search for new feature
    matches among the unmatched features of im1 using epipolar-guided brute-force
    descriptor matching.  Only features in im1 that have no existing match in
    *rmatches* are searched, and matches to im2 features already present in
    *rmatches* are discarded to keep the result free of duplicates.

    Args:
        im1, im2: Image names.
        camera1, camera2: Camera models for each image.
        p1, p2: Feature positions (in the masked feature space) for each image.
        rmatches: Existing robust matches as (N, 2) int array (indices into p1/p2).
        pose: Relative pose of camera2 w.r.t. camera1 (R_cam_to_world, origin),
            as returned by ``robust_match_calibrated``.
        data: Dataset.
        overriden_config: Merged configuration dictionary.

    Returns:
        New matches as an (M, 2) int array of indices into p1/p2.  Returns an
        empty (0, 2) array when no new matches are found.
    """
    empty: NDArray = np.empty((0, 2), dtype=int)

    segmentation_in_descriptor = overriden_config["matching_use_segmentation"]
    features_data1 = feature_loader.instance.load_all_data(
        data, im1, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    features_data2 = feature_loader.instance.load_all_data(
        data, im2, masked=True, segmentation_in_descriptor=segmentation_in_descriptor
    )
    if (
        features_data1 is None
        or features_data2 is None
        or features_data1.descriptors is None
        or features_data2.descriptors is None
    ):
        return empty

    d1 = features_data1.descriptors
    d2 = features_data2.descriptors

    # Determine which im1 features still lack a match
    matched_idx1: Set[int] = set(rmatches[:, 0].tolist()) if len(rmatches) > 0 else set()
    unmatched_idx1 = np.array(
        [i for i in range(len(p1)) if i not in matched_idx1], dtype=int
    )
    if len(unmatched_idx1) == 0:
        return empty

    # Track already-matched im2 features to avoid duplicate assignments
    matched_idx2: Set[int] = set(rmatches[:, 1].tolist()) if len(rmatches) > 0 else set()

    # Compute unit bearings for unmatched im1 features and all im2 features
    b1_unmatched = camera1.pixel_bearing_many(p1[unmatched_idx1, :2].copy())
    b2_all = camera2.pixel_bearing_many(p2[:, :2].copy())

    # Build epipolar consistency mask (len(unmatched_idx1) × len(p2))
    debug_label = (
        "post"
        if overriden_config.get("debug_epipolar_images", False)
        else None
    )
    epipolar_mask = compute_inliers_bearing_epipolar(
        b1_unmatched,
        b2_all,
        pose,
        overriden_config["guided_matching_threshold"],
        debug_label=debug_label,
        camera1=camera1,
        camera2=camera2,
        p1=p1[unmatched_idx1],
        p2=p2,
        data=data,
        im1=im1,
        im2=im2,
    )

    # Brute-force descriptor matching restricted to epipolar candidates
    d1_unmatched = d1[unmatched_idx1]
    new_local_matches = match_brute_force(d1_unmatched, d2, overriden_config, epipolar_mask)

    if not new_local_matches:
        return empty

    # Map local unmatched indices back to global indices in p1/p2,
    # and discard any match whose im2 feature is already used
    new_matches = [
        (int(unmatched_idx1[i]), int(j))
        for i, j in new_local_matches
        if j not in matched_idx2
    ]

    if not new_matches:
        return empty

    logger.debug(
        "Epipolar-guided post-matching {} and {}: {} new matches found".format(
            im1, im2, len(new_matches)
        )
    )
    return np.array(new_matches, dtype=int)


def match_words(
    f1: NDArray,
    words1: NDArray,
    f2: NDArray,
    words2: NDArray,
    config: Dict[str, Any],
) -> NDArray:
    """Match using words and apply Lowe's ratio filter.

    Args:
        f1: feature descriptors of the first image
        w1: the nth closest words for each feature in the first image
        f2: feature descriptors of the second image
        w2: the nth closest words for each feature in the second image
        config: config parameters
    """
    ratio = config["lowes_ratio"]
    num_checks = config["bow_num_checks"]
    return pyfeatures.match_using_words(f1, words1, f2, words2[:, 0], ratio, num_checks)


def match_words_symmetric(
    f1: NDArray,
    words1: NDArray,
    f2: NDArray,
    words2: NDArray,
    config: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Match using words in both directions and keep consistent matches.

    Args:
        f1: feature descriptors of the first image
        w1: the nth closest words for each feature in the first image
        f2: feature descriptors of the second image
        w2: the nth closest words for each feature in the second image
        config: config parameters
    """
    matches_ij = match_words(f1, words1, f2, words2, config)
    matches_ji = match_words(f2, words2, f1, words1, config)
    matches_ij = [(a, b) for a, b in matches_ij]
    matches_ji = [(b, a) for a, b in matches_ji]

    return list(set(matches_ij).intersection(set(matches_ji)))


def match_flann(
    index: cv2.flann_Index, f2: NDArray, config: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """Match using FLANN and apply Lowe's ratio filter.

    Args:
        index: flann index if the first image
        f2: feature descriptors of the second image
        config: config parameters
    """
    search_params = dict(checks=config["flann_checks"])
    results, dists = index.knnSearch(f2, 2, params=search_params)  # pyre-ignore[16]
    squared_ratio = config["lowes_ratio"] ** 2  # Flann returns squared L2 distances
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    return list(zip(results[good, 0], good.nonzero()[0]))


def match_flann_symmetric(
    fi: NDArray,
    indexi: cv2.flann_Index,
    fj: NDArray,
    indexj: cv2.flann_Index,
    config: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Match using FLANN in both directions and keep consistent matches.

    Args:
        fi: feature descriptors of the first image
        indexi: flann index if the first image
        fj: feature descriptors of the second image
        indexj: flann index of the second image
        config: config parameters
        maskij: optional boolean mask of len(i descriptors) x len(j descriptors)
    """
    matches_ij = [(a, b) for a, b in match_flann(indexi, fj, config)]
    matches_ji = [(b, a) for a, b in match_flann(indexj, fi, config)]

    return list(set(matches_ij).intersection(set(matches_ji)))


def match_brute_force(
    f1: NDArray,
    f2: NDArray,
    config: Dict[str, Any],
    maskij: Optional[NDArray] = None,
) -> List[Tuple[int, int]]:
    """Brute force matching and Lowe's ratio filtering.

    Args:
        f1: feature descriptors of the first image
        f2: feature descriptors of the second image
        config: config parameters
        maskij: optional boolean mask of len(i descriptors) x len(j descriptors)
    """
    assert f1.dtype.type == f2.dtype.type
    if f1.dtype.type == np.uint8:
        matcher_type = "BruteForce-Hamming"
    else:
        matcher_type = "BruteForce"
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    matcher.add([f2])
    if maskij is not None:
        matches = matcher.knnMatch(f1, k=2, masks=np.array([maskij]).astype(np.uint8))
    else:
        matches = matcher.knnMatch(f1, k=2)

    ratio = config["lowes_ratio"]
    good_matches = []
    for match in matches:
        if match and len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return [(mm.queryIdx, mm.trainIdx) for mm in good_matches]


def match_brute_force_symmetric(
    fi: NDArray,
    fj: NDArray,
    config: Dict[str, Any],
    maskij: Optional[NDArray] = None,
) -> List[Tuple[int, int]]:
    """Match with brute force in both directions and keep consistent matches.

    Args:
        fi: feature descriptors of the first image
        fj: feature descriptors of the second image
        config: config parameters
        maskij: optional boolean mask of len(i descriptors) x len(j descriptors)
    """
    matches_ij = [(a, b) for a, b in match_brute_force(fi, fj, config, maskij)]
    maskijT = maskij.T if maskij is not None else None
    matches_ji = [(b, a) for a, b in match_brute_force(fj, fi, config, maskijT)]

    return list(set(matches_ij).intersection(set(matches_ji)))


def robust_match_fundamental(
    p1: NDArray,
    p2: NDArray,
    matches: NDArray,
    config: Dict[str, Any],
) -> Tuple[NDArray, NDArray]:
    """Filter matches by estimating the Fundamental matrix via RANSAC."""
    if len(matches) < 8:
        return np.array([]), np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()

    FM_RANSAC = cv2.FM_RANSAC if context.OPENCV3 else cv2.cv.CV_FM_RANSAC
    threshold = config["robust_matching_threshold"]
    F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, threshold, 0.9999)
    inliers = mask.ravel().nonzero()

    if F is None or F[2, 2] == 0.0:
        return F, np.array([])

    return F, matches[inliers]


def compute_inliers_bearings(
    b1: NDArray,
    b2: NDArray,
    R: NDArray,
    t: NDArray,
    threshold: float = 0.01,
) -> List[bool]:
    """Compute points that can be triangulated.

    Args:
        b1, b2: Bearings in the two images.
        R, t: Rotation and translation from the second image to the first.
              That is the convention and the opposite of many
              functions in this module.
        threshold: max reprojection error in radians.
    Returns:
        array: Array of boolean indicating inliers/outliers
    """
    p = pygeometry.triangulate_two_bearings_midpoint_many(b1, b2, R, t)

    good_idx = [i for i in range(len(p)) if p[i][0]]
    points = np.array([p[i][1] for i in range(len(p)) if p[i][0]])

    inliers = [False] * len(b1)
    if len(points) < 1:
        return inliers

    br1 = points.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]
    br2 = R.T.dot((points - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = np.linalg.norm(br1 - b1[good_idx], axis=1) < threshold
    ok2 = np.linalg.norm(br2 - b2[good_idx], axis=1) < threshold
    is_ok = ok1 * ok2

    for i, ok in enumerate(is_ok):
        inliers[good_idx[i]] = ok
    return inliers


def _debug_epipolar(
    label: str,
    b1: NDArray,
    b2: NDArray,
    pose: pygeometry.Pose,
    threshold: float,
    angle_error: NDArray,
    mask: NDArray,
    camera1: Optional[pygeometry.Camera],
    camera2: Optional[pygeometry.Camera],
    p1: Optional[NDArray],
    p2: Optional[NDArray],
    data: Optional[DataSetBase],
    im1: Optional[str],
    im2: Optional[str],
) -> None:
    """Save epipolar debug images and log diagnostics for a pair of images.

    Draws:
      1. A text console summary (always, via logger).
      2. A side-by-side image where keypoints are colored green (passes
         threshold) or red (fails), with the epipole projected onto each image
         and a heat-map column showing the per-pair angle-error distribution.

    Args:
        label:        Short human-readable label (e.g. "guided" / "post").
        b1, b2:       Unit bearing arrays (N1×3, N2×3) in camera coordinates.
        pose:         Relative pose (camera2 w.r.t. camera1).
        threshold:    Epipolar-angle threshold in radians.
        angle_error:  Pre-computed N1×N2 angle-error matrix.
        mask:         Boolean mask derived from angle_error < threshold.
        camera1/2:    Camera models (may be None if not available).
        p1/p2:        Pixel coordinates of the features (may be None).
        data:         Dataset (may be None – disables image saving).
        im1/im2:      Image names (may be None – disables image saving).
    """
    n1, n2 = b1.shape[0], b2.shape[0]

    # --- Sanity checks ---------------------------------------------------------
    norms1 = np.linalg.norm(b1, axis=1)
    norms2 = np.linalg.norm(b2, axis=1)
    R = pose.get_R_cam_to_world()
    t = pose.get_origin()
    t_norm = float(np.linalg.norm(t))
    eye_err = float(np.linalg.norm(R @ R.T - np.eye(3)))

    logger.debug(
        "[epipolar debug %s | %s <-> %s]  "
        "n1=%d  n2=%d  "
        "bearing norms: b1 [%.4f, %.4f]  b2 [%.4f, %.4f]  "
        "R orthogonality error: %.2e  |t|=%.4f  "
        "threshold=%.4f rad  "
        "mask density: %.1f%%  "
        "angle_error min/median/max: %.4f / %.4f / %.4f rad",
        label,
        im1 or "?",
        im2 or "?",
        n1,
        n2,
        float(norms1.min()),
        float(norms1.max()),
        float(norms2.min()),
        float(norms2.max()),
        eye_err,
        t_norm,
        threshold,
        100.0 * float(mask.sum()) / max(mask.size, 1),
        float(angle_error.min()),
        float(np.median(angle_error)),
        float(angle_error.max()),
    )

    if t_norm < 1e-9:
        logger.warning(
            "[epipolar debug %s] Translation is near-zero (|t|=%.2e). "
            "Epipolar constraint degenerate – all bearings may pass.",
            label,
            t_norm,
        )

    # --- Image saving ----------------------------------------------------------
    if data is None or im1 is None or im2 is None:
        return
    if not hasattr(data, "data_path"):
        return
    if p1 is None or p2 is None or camera1 is None or camera2 is None:
        return

    try:
        img1 = data.load_image(im1, grayscale=False)
        img2 = data.load_image(im2, grayscale=False)
    except Exception as exc:
        logger.debug("[epipolar debug] Could not load images: %s", exc)
        return

    def to_bgr(img: NDArray) -> NDArray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img.copy()

    vis1 = to_bgr(img1)
    vis2 = to_bgr(img2)
    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]

    # Project epipole onto image1: the epipole is the projection of the
    # translation vector (origin of camera2 in camera1 frame).
    def _project_point(cam: pygeometry.Camera, bearing: NDArray, w: int, h: int) -> Optional[Tuple[int, int]]:
        """Back-project a bearing to pixel coordinates."""
        try:
            pts = cam.project_many(bearing.reshape(1, 3))
            if pts is None or len(pts) == 0:
                return None
            # pts are in normalized image coordinates [-0.5, 0.5]
            px = int((pts[0, 0] + 0.5) * w)
            py = int((pts[0, 1] + 0.5) * h)
            return px, py
        except Exception:
            return None

    # Bearing towards the epipole in camera1 = direction of t (translation).
    t_bearing1 = t / (np.linalg.norm(t) + 1e-12)
    ep1 = _project_point(camera1, t_bearing1, w1, h1)
    # Epipole in camera2 = projection of camera1's origin = -R^T * t in cam2.
    R_world_to_cam2 = R.T  # R_cam_to_world^T = R_world_to_cam2
    t_in_cam2 = -R_world_to_cam2 @ t
    t_bearing2 = t_in_cam2 / (np.linalg.norm(t_in_cam2) + 1e-12)
    ep2 = _project_point(camera2, t_bearing2, w2, h2)

    # Draw keypoints for im1 that appear in b1 (subset or all features).
    # p1 and p2 hold pixel coordinates in normalized coords (OpenSfM convention).
    # Color each keypoint by the *fraction* of compatible partners in the other image:
    #   green  = few compatible partners  (tight epipolar constraint, good)
    #   yellow = moderate fraction        (partial constraint)
    #   red    = many compatible partners (loose/degenerate constraint, suspicious)
    # This makes the visualization meaningful even at low overall mask densities:
    # mask.any(axis=1) would make nearly every feature green whenever n2 is large,
    # because even at 0.5% mask density each feature has ~5 compatible partners on
    # average and the probability of having zero is only ~0.7%.
    def _frac_to_bgr(frac: float) -> Tuple[int, int, int]:
        """Map fraction-of-compatible-partners [0,1] to a BGR colour.

        0.0 -> pure green  (tight constraint, desirable)
        0.5 -> yellow
        1.0 -> pure red    (every partner compatible, constraint useless)
        """
        f = float(np.clip(frac, 0.0, 1.0))
        if f <= 0.5:
            # green -> yellow
            r = int(2 * f * 255)
            g = 200
        else:
            # yellow -> red
            r = 200
            g = int((1.0 - 2 * (f - 0.5)) * 200)
        return (0, g, r)  # BGR

    def draw_kpts(vis: NDArray, pts_norm: NDArray, per_feature_frac: NDArray, w: int, h: int) -> None:
        for idx in range(len(pts_norm)):
            px = int((float(pts_norm[idx, 0]) + 0.5) * w)
            py = int((float(pts_norm[idx, 1]) + 0.5) * h)
            color = _frac_to_bgr(per_feature_frac[idx])
            cv2.circle(vis, (px, py), 4, color, -1)

    # Fraction of compatible partners for each feature (0 = fully constrained,
    # 1 = every candidate in the other image is compatible = constraint useless).
    frac1 = mask.mean(axis=1).astype(float)  # shape (n1,)
    frac2 = mask.mean(axis=0).astype(float)  # shape (n2,)

    # Log the constraint-quality summary so it is visible without images.
    logger.debug(
        "[epipolar debug %s] compatible-partner fractions "
        "im1: median=%.3f  90th=%.3f  max=%.3f  "
        "im2: median=%.3f  90th=%.3f  max=%.3f",
        label,
        float(np.median(frac1)),
        float(np.percentile(frac1, 90)),
        float(frac1.max()),
        float(np.median(frac2)),
        float(np.percentile(frac2, 90)),
        float(frac2.max()),
    )

    draw_kpts(vis1, p1[:n1], frac1, w1, h1)
    draw_kpts(vis2, p2[:n2], frac2, w2, h2)

    # Draw epipole cross
    for ep, vis, w, h in [(ep1, vis1, w1, h1), (ep2, vis2, w2, h2)]:
        if ep is not None:
            cx, cy = ep
            cv2.drawMarker(
                vis,
                (cx, cy),
                (255, 0, 255),
                cv2.MARKER_CROSS,
                30,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "epipole",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        # Draw legend at the bottom of each image panel
        cv2.putText(
            vis,
            "green=tight  red=loose constraint",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Build angle-error heat-map column (max 256 px wide)
    hmap_w = min(256, max(64, angle_error.shape[1]))
    hmap_h = max(h1, h2)
    # Normalize to [0, 255] using 3x threshold as upper bound
    ae_scaled = np.clip(angle_error / (3.0 * threshold + 1e-12), 0.0, 1.0)
    ae_thumb = cv2.resize(
        (ae_scaled * 255).astype(np.uint8),
        (hmap_w, hmap_h),
        interpolation=cv2.INTER_NEAREST,
    )
    ae_color = cv2.applyColorMap(ae_thumb, cv2.COLORMAP_JET)
    # Draw threshold line
    thresh_x = int(hmap_w / 3.0)  # corresponds to threshold / (3*threshold)=1/3
    cv2.line(ae_color, (thresh_x, 0), (thresh_x, hmap_h - 1), (255, 255, 255), 2)
    cv2.putText(
        ae_color,
        "angle err",
        (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        ae_color,
        "white=thresh",
        (4, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Resize images to the same height
    target_h = max(h1, h2, hmap_h)

    def _resize_h(img: NDArray, target: int) -> NDArray:
        h, w = img.shape[:2]
        if h == target:
            return img
        scale = target / h
        return cv2.resize(img, (int(w * scale), target))

    vis1_r = _resize_h(vis1, target_h)
    vis2_r = _resize_h(vis2, target_h)
    ae_r = _resize_h(ae_color, target_h)

    combined = np.hstack([vis1_r, vis2_r, ae_r])

    # Add header
    header_h = 36
    header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{label}: {im1} <-> {im2}  mask={100*float(mask.sum())/max(mask.size,1):.1f}%  |t|={t_norm:.4f}  thr={threshold:.4f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 200),
        1,
        cv2.LINE_AA,
    )
    combined = np.vstack([header, combined])

    debug_dir = os.path.join(
        data.data_path,
        "debug",
        "epipolar",
        f"{im1}__{im2}",
    )
    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, f"{label}.jpg")
    cv2.imwrite(out_path, combined)
    logger.debug("[epipolar debug] saved %s", out_path)


def compute_inliers_bearing_epipolar(
    b1: NDArray,
    b2: NDArray,
    pose: pygeometry.Pose,
    threshold: float,
    debug_label: Optional[str] = None,
    camera1: Optional[pygeometry.Camera] = None,
    camera2: Optional[pygeometry.Camera] = None,
    p1: Optional[NDArray] = None,
    p2: Optional[NDArray] = None,
    data: Optional[DataSetBase] = None,
    im1: Optional[str] = None,
    im2: Optional[str] = None,
) -> NDArray:
    """Compute mask of epipolarly consistent bearings, given two lists of bearings

    Args:
        b1, b2: Bearings in the two images. Expected to be normalized.
        pose: Pose of the second image wrt. the first one (relative pose)
        threshold: max reprojection error in radians.
        debug_label: If not None, call ``_debug_epipolar`` with this label.
        camera1/2: Camera models, needed for debug image projection.
        p1/p2: Feature pixel positions (normalized coords), needed for debug images.
        data: Dataset, needed for loading source images for debug output.
        im1/im2: Image names, needed for debug output paths.
    Returns:
        array: Matrix of boolean indicating inliers/outliers
    """
    symmetric_angle_error = pygeometry.epipolar_angle_two_bearings_many(
        b1.astype(np.float32),
        b2.astype(np.float32),
        pose.get_R_cam_to_world(),
        pose.get_origin(),
    )
    mask = symmetric_angle_error < threshold
    if debug_label is not None:
        _debug_epipolar(
            debug_label,
            b1,
            b2,
            pose,
            threshold,
            symmetric_angle_error,
            mask,
            camera1,
            camera2,
            p1,
            p2,
            data,
            im1,
            im2,
        )
    return mask


def robust_match_calibrated(
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    matches: NDArray,
    config: Dict[str, Any],
) -> Tuple[NDArray, Optional[pygeometry.Pose]]:
    """Filter matches by estimating the Essential matrix via RANSAC.

    Returns:
        Tuple of (inlier matches, estimated relative pose of camera2 w.r.t.
        camera1).  The pose is None when estimation fails.
    """

    if len(matches) < 8:
        return np.array([]), None

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    threshold = config["robust_matching_calib_threshold"]
    T = multiview.relative_pose_ransac(b1, b2, threshold, 1000, 0.999)

    for relax in [4, 2, 1]:
        inliers = compute_inliers_bearings(b1, b2, T[:, :3], T[:, 3], relax * threshold)
        if np.sum(inliers) < 8:
            return np.array([]), None
        iterations = config["five_point_refine_match_iterations"]
        T = multiview.relative_pose_optimize_nonlinear(
            b1[inliers], b2[inliers], T[:3, 3], T[:3, :3], iterations
        )

    inliers = compute_inliers_bearings(b1, b2, T[:, :3], T[:, 3], threshold)

    # Build a Pose from T.  By the convention in compute_inliers_bearings,
    # T[:3, :3] is R_cam2_to_cam1 (rotation from camera2 to camera1 frame)
    # and T[:3, 3] is the origin of camera2 expressed in camera1 frame.
    # set_from_cam_to_world stores R and t as [R_cw | t_cw] in the
    # cam_to_world matrix, so get_R_cam_to_world() == T[:3, :3] and
    # get_origin() == T[:3, 3], which is what compute_inliers_bearing_epipolar
    # expects.
    pose = pygeometry.Pose()
    pose.set_from_cam_to_world(T[:3, :3], T[:3, 3])

    return matches[inliers], pose


def robust_match(
    p1: NDArray,
    p2: NDArray,
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
    matches: NDArray,
    config: Dict[str, Any],
) -> Tuple[NDArray, Optional[pygeometry.Pose]]:
    """Filter matches by fitting a geometric model.

    If cameras are perspective without distortion, then the Fundamental
    matrix is used.  Otherwise, we use the Essential matrix.

    Returns:
        Tuple of (inlier matches, optional relative pose).  The pose is
        available only for the Essential-matrix (calibrated) path; it is
        None for the Fundamental-matrix path.
    """
    if (
        camera1.projection_type in ["perspective", "brown"]
        and camera1.k1 == 0.0
        and camera1.k2 == 0.0
        and camera2.projection_type in ["perspective", "brown"]
        and camera2.k1 == 0.0
        and camera2.k2 == 0.0
    ):
        return robust_match_fundamental(p1, p2, matches, config)[1], None
    else:
        return robust_match_calibrated(p1, p2, camera1, camera2, matches, config)


def unfilter_matches(matches: NDArray, m1: NDArray, m2: NDArray) -> NDArray:
    """Given matches and masking arrays, get matches with un-masked indexes."""
    i1 = np.flatnonzero(m1)
    i2 = np.flatnonzero(m2)
    return np.array([(i1[match[0]], i2[match[1]]) for match in matches])


def apply_adhoc_filters(
    data: DataSetBase,
    matches: List[Tuple[int, int]],
    im1: str,
    camera1: pygeometry.Camera,
    p1: NDArray,
    im2: str,
    camera2: pygeometry.Camera,
    p2: NDArray,
) -> List[Tuple[int, int]]:
    """Apply a set of filters functions defined further below
    for removing static data in images.

    """
    matches = _non_static_matches(p1, p2, matches)
    matches = _not_on_pano_poles_matches(p1, p2, matches, camera1, camera2)
    matches = _not_on_vermont_watermark(p1, p2, matches, im1, im2, data)
    matches = _not_on_blackvue_watermark(p1, p2, matches, im1, im2, data)
    return matches


def _non_static_matches(
    p1: NDArray, p2: NDArray, matches: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Remove matches with same position in both images.

    That should remove matches on that are likely belong to rig occluders,
    watermarks or dust, but not discard entirely static images.
    """
    threshold = 0.001
    res = []
    for match in matches:
        d = p1[match[0]] - p2[match[1]]
        if d[0] ** 2 + d[1] ** 2 >= threshold**2:
            res.append(match)

    static_ratio_threshold = 0.85
    static_ratio_removed = 1 - len(res) / max(len(matches), 1)
    if static_ratio_removed > static_ratio_threshold:
        return matches
    else:
        return res


def _not_on_pano_poles_matches(
    p1: NDArray,
    p2: NDArray,
    matches: List[Tuple[int, int]],
    camera1: pygeometry.Camera,
    camera2: pygeometry.Camera,
) -> List[Tuple[int, int]]:
    """Remove matches for features that are too high or to low on a pano.

    That should remove matches on the sky and and carhood part of panoramas
    """
    min_lat = -0.125
    max_lat = 0.125
    is_pano1 = pygeometry.Camera.is_panorama(camera1.projection_type)
    is_pano2 = pygeometry.Camera.is_panorama(camera2.projection_type)
    if is_pano1 or is_pano2:
        res = []
        for match in matches:
            if (not is_pano1 or min_lat < p1[match[0]][1] < max_lat) and (
                not is_pano2 or min_lat < p2[match[1]][1] < max_lat
            ):
                res.append(match)
        return res
    else:
        return matches


def _not_on_vermont_watermark(
    p1: NDArray,
    p2: NDArray,
    matches: List[Tuple[int, int]],
    im1: str,
    im2: str,
    data: DataSetBase,
) -> List[Tuple[int, int]]:
    """Filter Vermont images watermark."""
    meta1 = data.load_exif(im1)
    meta2 = data.load_exif(im2)

    if meta1["make"] == "VTrans_Camera" and meta1["model"] == "VTrans_Camera":
        matches = [m for m in matches if _vermont_valid_mask(p1[m[0]])]
    if meta2["make"] == "VTrans_Camera" and meta2["model"] == "VTrans_Camera":
        matches = [m for m in matches if _vermont_valid_mask(p2[m[1]])]
    return matches


def _vermont_valid_mask(p: NDArray) -> bool:
    """Check if pixel inside the valid region.

    Pixel coord Y should be larger than 50.
    In normalized coordinates y > (50 - h / 2) / w
    """
    return p[1] > -0.255


def _not_on_blackvue_watermark(
    p1: NDArray,
    p2: NDArray,
    matches: List[Tuple[int, int]],
    im1: str,
    im2: str,
    data: DataSetBase,
) -> List[Tuple[int, int]]:
    """Filter Blackvue's watermark."""
    meta1 = data.load_exif(im1)
    meta2 = data.load_exif(im2)

    if meta1["make"].lower() == "blackvue":
        matches = [m for m in matches if _blackvue_valid_mask(p1[m[0]])]
    if meta2["make"].lower() == "blackvue":
        matches = [m for m in matches if _blackvue_valid_mask(p2[m[1]])]
    return matches


def _blackvue_valid_mask(p: NDArray) -> bool:
    """Check if pixel inside the valid region.

    Pixel coord Y should be smaller than h - 70.
    In normalized coordinates y < (h - 70 - h / 2) / w,
    with h = 2160 and w = 3840
    """
    return p[1] < 0.263
