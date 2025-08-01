# pyre-strict

from types import ModuleType
from typing import List

from . import (
    align_submodels,
    bundle,
    compute_depthmaps,
    compute_statistics,
    create_rig,
    create_submodels,
    create_tracks,
    detect_features,
    export_bundler,
    export_colmap,
    export_geocoords,
    export_openmvs,
    export_ply,
    export_pmvs,
    export_report,
    export_visualsfm,
    extend_reconstruction,
    extract_metadata,
    match_features,
    mesh,
    reconstruct,
    reconstruct_from_prior,
    undistort,
)
from .command_runner import command_runner


opensfm_commands: List[ModuleType] = [
    extract_metadata,
    detect_features,
    match_features,
    create_rig,
    create_tracks,
    reconstruct,
    reconstruct_from_prior,
    bundle,
    mesh,
    undistort,
    compute_depthmaps,
    compute_statistics,
    export_ply,
    export_openmvs,
    export_visualsfm,
    export_pmvs,
    export_bundler,
    export_colmap,
    export_geocoords,
    export_report,
    extend_reconstruction,
    create_submodels,
    align_submodels,
]
