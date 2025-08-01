# pyre-strict
import logging
import typing as t
from typing import cast, Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from opensfm import pymap
from opensfm.dataset_base import DataSetBase
from opensfm.pymap import TracksManager
from opensfm.unionfind import UnionFind


logger: logging.Logger = logging.getLogger(__name__)


def load_features(
    dataset: DataSetBase, images: List[str]
) -> Tuple[
    Dict[str, NDArray[np.float64]],
    Dict[str, NDArray[np.int32]],
    Dict[str, NDArray[np.int32]],
    Dict[str, NDArray[np.int32]],
    Dict[str, NDArray[np.float64]],
]:
    logging.info("reading features")
    features = {}
    colors = {}
    segmentations = {}
    instances = {}
    depths = {}
    for im in images:
        features_data = dataset.load_features(im)

        if not features_data:
            continue

        features[im] = features_data.points[:, :3]
        colors[im] = features_data.colors

        semantic_data = features_data.semantic
        if semantic_data:
            segmentations[im] = semantic_data.segmentation
            if semantic_data.has_instances():
                instances[im] = semantic_data.instances

        depth_data = features_data.depths
        if depth_data is not None:
            depths[im] = depth_data

    return features, colors, segmentations, instances, depths


def load_matches(
    dataset: DataSetBase, images: List[str]
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    matches = {}
    for im1 in images:
        try:
            im1_matches = dataset.load_matches(im1)
        except IOError:
            continue
        for im2 in im1_matches:
            if im2 in images:
                matches[im1, im2] = im1_matches[im2]
    return matches


def create_tracks_manager(
    features: Dict[str, NDArray[np.float64]],
    colors: Dict[str, NDArray[np.int32]],
    segmentations: Dict[str, NDArray[np.int32]],
    instances: Dict[str, NDArray[np.int32]],
    matches: Dict[Tuple[str, str], List[Tuple[int, int]]],
    min_length: int,
    depths: Dict[str, NDArray[np.float64]],
    depth_is_radial: bool = True,
    depth_std_deviation: float = 1.0,
) -> TracksManager:
    """Link matches into tracks."""
    logger.debug("Merging features onto tracks")
    uf = UnionFind()
    for im1, im2 in matches:
        for f1, f2 in matches[im1, im2]:
            uf.union((im1, f1), (im2, f2))

    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    tracks = [t for t in sets.values() if _good_track(t, min_length)]

    NO_VALUE = pymap.Observation.NO_SEMANTIC_VALUE
    tracks_manager = pymap.TracksManager()
    num_observations = 0
    num_depth_priors = 0
    for track_id, track in enumerate(tracks):
        for image, featureid in track:
            if image not in features:
                continue
            x, y, s = features[image][featureid]
            r, g, b = colors[image][featureid]
            segmentation = (
                int(segmentations[image][featureid])
                if image in segmentations
                else NO_VALUE
            )
            instance = (
                int(instances[image][featureid]) if image in instances else NO_VALUE
            )

            obs = pymap.Observation(
                x,
                y,
                s,
                int(r),
                int(g),
                int(b),
                featureid,
                segmentation,
                instance,
            )
            if image in depths:
                depth_value = depths[image][featureid]
                if not np.isnan(depth_value) and not np.isinf(depth_value):
                    std = max(
                        depth_std_deviation * depth_value,  # pyre-ignore
                        depth_std_deviation,
                    )
                    obs.depth_prior = pymap.Depth(
                        value=depth_value,  # pyre-ignore
                        std_deviation=std,
                        is_radial=depth_is_radial,
                    )
                    num_depth_priors += 1
            tracks_manager.add_observation(image, str(track_id), obs)
            num_observations += 1
    logger.info(
        f"{len(tracks)} tracks, {num_observations} observations,"
        f" {num_depth_priors} depth priors added to TracksManager"
    )
    return tracks_manager


def common_tracks(
    tracks_manager: pymap.TracksManager, im1: str, im2: str
) -> Tuple[List[str], NDArray[np.float64], NDArray[np.float64]]:
    """List of tracks observed in both images.

    Args:
        tracks_manager: tracks manager
        im1: name of the first image
        im2: name of the second image

    Returns:
        tuple: tracks, feature from first image, feature from second image
    """
    t1 = tracks_manager.get_shot_observations(im1)
    t2 = tracks_manager.get_shot_observations(im2)
    tracks, p1, p2 = [], [], []
    for track, obs in t1.items():
        if track in t2:
            p1.append(obs.point)
            p2.append(t2[track].point)
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


TPairTracks = Tuple[List[str], NDArray[np.float64], NDArray[np.float64]]


def all_common_tracks_with_features(
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> Dict[Tuple[str, str], TPairTracks]:
    tracks = all_common_tracks(
        tracks_manager, include_features=True, min_common=min_common
    )
    return cast(Dict[Tuple[str, str], TPairTracks], tracks)


def all_common_tracks_without_features(
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> Dict[Tuple[str, str], List[str]]:
    tracks = all_common_tracks(
        tracks_manager, include_features=False, min_common=min_common
    )
    return cast(Dict[Tuple[str, str], List[str]], tracks)


def all_common_tracks(
    tracks_manager: pymap.TracksManager,
    include_features: bool = True,
    min_common: int = 50,
) -> Dict[Tuple[str, str], t.Union[TPairTracks, List[str]]]:
    """List of tracks observed by each image pair.

    Args:
        tracks_manager: tracks manager
        include_features: whether to include the features from the images
        min_common: the minimum number of tracks the two images need to have
            in common

    Returns:
        tuple: im1, im2 -> tuple: tracks, features from first image, features
        from second image
    """
    common_tracks = {}
    for (im1, im2), size in tracks_manager.get_all_pairs_connectivity().items():
        if size < min_common:
            continue

        tuples = tracks_manager.get_all_common_observations(im1, im2)
        if include_features:
            common_tracks[im1, im2] = (
                [v for v, _, _ in tuples],
                np.array([p.point for _, p, _ in tuples]),
                np.array([p.point for _, _, p in tuples]),
            )
        else:
            common_tracks[im1, im2] = [v for v, _, _ in tuples]
    return common_tracks


def _good_track(track: List[Tuple[str, int]], min_length: int) -> bool:
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True


def as_weighted_graph(tracks_manager: pymap.TracksManager) -> nx.Graph:
    """Return the tracks manager as a weighted graph
    having shots a snodes and weighted by the # of
    common tracks between two nodes.
    """
    images = tracks_manager.get_shot_ids()
    image_graph = nx.Graph()
    for im in images:
        image_graph.add_node(im)
    for k, v in tracks_manager.get_all_pairs_connectivity().items():
        image_graph.add_edge(k[0], k[1], weight=v)
    return image_graph


def as_graph(tracks_manager: pymap.TracksManager) -> nx.Graph:
    """Return the tracks manager as a bipartite graph (legacy)."""
    tracks = tracks_manager.get_track_ids()
    images = tracks_manager.get_shot_ids()

    graph = nx.Graph()
    for track_id in tracks:
        graph.add_node(track_id, bipartite=1)
    for shot_id in images:
        graph.add_node(shot_id, bipartite=0)
    for track_id in tracks:
        for im, obs in tracks_manager.get_track_observations(track_id).items():
            graph.add_edge(
                im,
                track_id,
                feature=obs.point,
                feature_scale=obs.scale,
                feature_id=obs.id,
                feature_color=obs.color,
                feature_segmentation=obs.segmentation,
                feature_instance=obs.instance,
            )
    return graph
