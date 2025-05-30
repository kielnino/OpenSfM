#include <foundation/optional.h>
#include <foundation/python_types.h>
#include <foundation/types.h>
#include <geometry/camera.h>
#include <geometry/pose.h>
#include <map/dataviews.h>
#include <map/defines.h>
#include <map/ground_control_points.h>
#include <map/landmark.h>
#include <map/map.h>
#include <map/pybind_utils.h>
#include <map/rig.h>
#include <map/shot.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <typeinfo>
namespace py = pybind11;

template <typename T>
void DeclareShotMeasurement(py::module &m, const std::string &type_name) {
  using SM = foundation::OptionalValue<T>;

  std::string class_name = std::string("ShotMeasurement") + type_name;

  py::class_<SM>(m, class_name.c_str())
      .def(py::init<>())
      .def_property_readonly("has_value", &SM::HasValue)
      .def_property("value", py::overload_cast<>(&SM::Value, py::const_),
                    &SM::SetValue)
      .def("reset", &SM::Reset)
      .def(py::pickle(
          [](const SM &sm) {
            return py::make_tuple(sm.HasValue(), sm.Value());
          },
          [](py::tuple p) {
            SM sm;
            const auto has_value = p[0].cast<bool>();
            if (has_value) {
              sm.SetValue(p[1].cast<T>());
            }
            return sm;
          }));
}
PYBIND11_MODULE(pymap, m) {
  py::module::import("opensfm.pygeometry");
  py::module::import("opensfm.pygeo");

  // Some initial definitions to resolve cyclic dependencies
  // Landmark <> Shot
  py::class_<map::Shot> shotCls(m, "Shot");
  // Landmark/Shot/...View <> Map
  py::class_<map::Map> mapCls(m, "Map");

  DeclareShotMeasurement<int>(m, "Int");
  DeclareShotMeasurement<double>(m, "Double");
  DeclareShotMeasurement<Vec3d>(m, "Vec3d");
  DeclareShotMeasurement<std::string>(m, "String");

  py::enum_<map::Map::ErrorType>(m, "ErrorType")
      .value("Pixel", map::Map::Pixel)
      .value("Normalized", map::Map::Normalized)
      .value("Angular", map::Map::Angular)
      .export_values();

  py::class_<map::Depth>(m, "Depth")
      .def(py::init<double, bool, double>(), py::arg("value"),
           py::arg("is_radial"), py::arg("std_deviation"))
      .def_readwrite("value", &map::Depth::value)
      .def_readwrite("is_radial", &map::Depth::is_radial)
      .def_readwrite("std_deviation", &map::Depth::std_deviation);

  py::class_<map::Observation>(m, "Observation")
      .def(py::init<double, double, double, int, int, int, int, int, int>(),
           py::arg("x"), py::arg("y"), py::arg("s"), py::arg("r"), py::arg("g"),
           py::arg("b"), py::arg("feature"),
           py::arg("segmentation") = map::Observation::NO_SEMANTIC_VALUE,
           py::arg("instance") = map::Observation::NO_SEMANTIC_VALUE)
      .def_readwrite("point", &map::Observation::point)
      .def_readwrite("scale", &map::Observation::scale)
      .def_readwrite("id", &map::Observation::feature_id)
      .def_readwrite("color", &map::Observation::color)
      .def_readwrite("segmentation", &map::Observation::segmentation_id)
      .def_readwrite("instance", &map::Observation::instance_id)
      .def_readwrite("depth_prior", &map::Observation::depth_prior)
      .def_readonly_static("NO_SEMANTIC_VALUE",
                           &map::Observation::NO_SEMANTIC_VALUE)
      .def(
          "copy",
          [](const map::Observation &to_copy) {
            map::Observation copy = to_copy;
            return copy;
          },
          py::return_value_policy::copy);

  py::class_<map::Landmark>(m, "Landmark")
      .def(py::init<const map::LandmarkId &, const Vec3d &>())
      .def_readonly("id", &map::Landmark::id_)
      .def_property("coordinates", &map::Landmark::GetGlobalPos,
                    &map::Landmark::SetGlobalPos)
      .def("get_observations", &map::Landmark::GetObservations,
           py::return_value_policy::reference_internal)
      .def("number_of_observations", &map::Landmark::NumberOfObservations)
      .def_property("reprojection_errors",
                    &map::Landmark::GetReprojectionErrors,
                    &map::Landmark::SetReprojectionErrors)
      .def_property("color", &map::Landmark::GetColor,
                    &map::Landmark::SetColor);

  py::class_<map::ShotMeasurements>(m, "ShotMeasurements")
      .def(py::init<>())
      .def_readwrite("gps_accuracy", &map::ShotMeasurements::gps_accuracy_)
      .def_readwrite("gps_position", &map::ShotMeasurements::gps_position_)
      .def_readwrite("orientation", &map::ShotMeasurements::orientation_)
      .def_readwrite("capture_time", &map::ShotMeasurements::capture_time_)
      .def_readwrite("gravity_down", &map::ShotMeasurements::gravity_down_)
      .def_readwrite("compass_angle", &map::ShotMeasurements::compass_angle_)
      .def_readwrite("compass_accuracy",
                     &map::ShotMeasurements::compass_accuracy_)
      .def_readwrite("opk_angles", &map::ShotMeasurements::opk_angles_)
      .def_readwrite("opk_accuracy", &map::ShotMeasurements::opk_accuracy_)
      .def_readwrite("sequence_key", &map::ShotMeasurements::sequence_key_)
      .def_property("attributes", &map::ShotMeasurements::GetAttributes,
                    &map::ShotMeasurements::SetAttributes)
      .def(py::pickle(
          [](const map::ShotMeasurements &s) {
            return py::make_tuple(
                s.gps_accuracy_, s.gps_position_, s.orientation_,
                s.capture_time_, s.gravity_down_, s.compass_angle_,
                s.compass_accuracy_, s.opk_angles_, s.opk_accuracy_,
                s.sequence_key_, s.GetAttributes());
          },
          [](py::tuple s) {
            map::ShotMeasurements sm;
            sm.gps_accuracy_ = s[0].cast<decltype(sm.gps_accuracy_)>();
            sm.gps_position_ = s[1].cast<decltype(sm.gps_position_)>();
            sm.orientation_ = s[2].cast<decltype(sm.orientation_)>();
            sm.capture_time_ = s[3].cast<decltype(sm.capture_time_)>();
            sm.gravity_down_ = s[4].cast<decltype(sm.gravity_down_)>();
            sm.compass_angle_ = s[5].cast<decltype(sm.compass_angle_)>();
            sm.compass_accuracy_ = s[6].cast<decltype(sm.compass_accuracy_)>();
            sm.opk_angles_ = s[7].cast<decltype(sm.opk_angles_)>();
            sm.opk_accuracy_ = s[8].cast<decltype(sm.opk_accuracy_)>();
            sm.sequence_key_ = s[9].cast<decltype(sm.sequence_key_)>();
            sm.GetMutableAttributes() = s[10].cast<decltype(sm.attributes_)>();
            return sm;
          }))
      .def(
          "__copy__",
          [](const map::ShotMeasurements &to_copy) {
            map::ShotMeasurements copy;
            copy.Set(to_copy);
            return copy;
          },
          py::return_value_policy::copy)

      .def("set", &map::ShotMeasurements::Set);

  py::class_<map::ShotMesh>(m, "ShotMesh")
      .def_property("faces", &map::ShotMesh::GetFaces, &map::ShotMesh::SetFaces)
      .def_property("vertices", &map::ShotMesh::GetVertices,
                    &map::ShotMesh::SetVertices);

  py::class_<map::RigCamera>(m, "RigCamera")
      .def(py::init<>())
      .def(py::init<const geometry::Pose &, const map::RigCameraId &>())
      .def_readwrite("id", &map::RigCamera::id)
      .def_readwrite("pose", &map::RigCamera::pose)
      // pickle support
      .def(py::pickle(
          [](const map::RigCamera &rc) {
            return py::make_tuple(rc.pose, rc.id);
          },
          [](py::tuple s) {
            return map::RigCamera(s[0].cast<geometry::Pose>(),
                                  s[1].cast<map::RigCameraId>());
          }));

  py::class_<map::RigInstance>(m, "RigInstance")
      .def(py::init<map::RigInstanceId>())
      .def_readwrite("id", &map::RigInstance::id)
      .def_property_readonly("shots",
                             py::overload_cast<>(&map::RigInstance::GetShots),
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "rig_cameras", py::overload_cast<>(&map::RigInstance::GetRigCameras),
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "rig_camera_ids",
          [](const map::RigInstance &ri) {
            std::map<map::ShotId, map::RigCameraId> rig_camera_ids;
            for (const auto &rig_camera : ri.GetRigCameras()) {
              rig_camera_ids[rig_camera.first] = rig_camera.second->id;
            }
            return rig_camera_ids;
          })
      .def_property_readonly("camera_ids",
                             [](const map::RigInstance &ri) {
                               std::map<map::ShotId, map::CameraId> camera_ids;
                               for (const auto &shot : ri.GetShots()) {
                                 camera_ids[shot.first] =
                                     shot.second->GetCamera()->id;
                               }
                               return camera_ids;
                             })
      .def("keys", &map::RigInstance::GetShotIDs)
      .def_property("pose", py::overload_cast<>(&map::RigInstance::GetPose),
                    &map::RigInstance::SetPose,
                    py::return_value_policy::reference_internal)
      .def("add_shot", &map::RigInstance::AddShot)
      .def("remove_shot", &map::RigInstance::RemoveShot)
      .def("update_instance_pose_with_shot",
           &map::RigInstance::UpdateInstancePoseWithShot)
      .def("update_rig_camera_pose", &map::RigInstance::UpdateRigCameraPose);

  shotCls
      .def(py::init<const map::ShotId &, const geometry::Camera &,
                    const geometry::Pose &>())
      .def_readonly("id", &map::Shot::id_)
      .def_readwrite("mesh", &map::Shot::mesh)
      .def_property("covariance", &map::Shot::GetCovariance,
                    &map::Shot::SetCovariance)
      .def_readwrite("merge_cc", &map::Shot::merge_cc)
      .def_readwrite("scale", &map::Shot::scale)
      .def_property_readonly("rig_instance", &map::Shot::GetRigInstance)
      .def_property_readonly("rig_camera", &map::Shot::GetRigCamera)
      .def_property_readonly("rig_instance_id", &map::Shot::GetRigInstanceId)
      .def_property_readonly("rig_camera_id", &map::Shot::GetRigCameraId)
      .def("set_rig", &map::Shot::SetRig)
      .def("get_observation", &map::Shot::GetObservation,
           py::return_value_policy::reference_internal)
      .def("get_valid_landmarks", &map::Shot::ComputeValidLandmarks)
      .def("remove_observation", &map::Shot::RemoveLandmarkObservation)
      .def_property("metadata",
                    py::overload_cast<>(&map::Shot::GetShotMeasurements),
                    &map::Shot::SetShotMeasurements,
                    py::return_value_policy::reference_internal)
      .def_property("pose", py::overload_cast<>(&map::Shot::GetPose),
                    &map::Shot::SetPose,
                    py::return_value_policy::reference_internal)
      .def_property_readonly("camera", &map::Shot::GetCamera,
                             py::return_value_policy::reference_internal)
      .def("get_landmark_observation", &map::Shot::GetLandmarkObservation,
           py::return_value_policy::reference_internal)
      .def("get_observation_landmark", &map::Shot::GetObservationLandmark,
           py::return_value_policy::reference_internal)
      .def("project", &map::Shot::Project)
      .def("project_many", &map::Shot::ProjectMany)
      .def("bearing", &map::Shot::Bearing)
      .def("bearing_many", &map::Shot::BearingMany);

  py::class_<map::GroundControlPointObservation>(
      m, "GroundControlPointObservation")
      .def(py::init())
      .def(py::init<const map::ShotId &, const Vec2d &>())
      .def_readwrite("shot_id", &map::GroundControlPointObservation::shot_id_)
      .def_readwrite("uid", &map::GroundControlPointObservation::uid_)
      .def_readwrite("projection",
                     &map::GroundControlPointObservation::projection_);

  py::enum_<map::GroundControlPointRole>(m, "GroundControlPointRole")
      .value("OPTIMIZATION", map::GroundControlPointRole::OPTIMIZATION)
      .value("METRICS_ONLY", map::GroundControlPointRole::METRICS_ONLY)
      .export_values();

  py::class_<map::GroundControlPoint>(m, "GroundControlPoint")
      .def(py::init())
      .def_readwrite("id", &map::GroundControlPoint::id_)
      .def_readwrite("survey_point_id",
                     &map::GroundControlPoint::survey_point_id_)
      .def_readwrite("has_altitude", &map::GroundControlPoint::has_altitude_)
      .def_readwrite("lla", &map::GroundControlPoint::lla_)
      .def_property("lla_vec", &map::GroundControlPoint::GetLlaVec3d,
                    &map::GroundControlPoint::SetLla)
      .def_readwrite("role", &map::GroundControlPoint::role_)
      .def_property("observations", &map::GroundControlPoint::GetObservations,
                    &map::GroundControlPoint::SetObservations)
      .def("add_observation", &map::GroundControlPoint::AddObservation);

  py::class_<map::TracksManager>(m, "TracksManager")
      .def(py::init())
      .def_static("instanciate_from_file",
                  &map::TracksManager::InstanciateFromFile,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("instanciate_from_string",
                  &map::TracksManager::InstanciateFromString,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("merge_tracks_manager",
                  &map::TracksManager::MergeTracksManager)
      .def("add_observation", &map::TracksManager::AddObservation)
      .def("remove_observation", &map::TracksManager::RemoveObservation)
      .def("num_shots", &map::TracksManager::NumShots)
      .def("num_tracks", &map::TracksManager::NumTracks)
      .def("get_shot_ids", &map::TracksManager::GetShotIds)
      .def("get_track_ids", &map::TracksManager::GetTrackIds)
      .def("get_observation", &map::TracksManager::GetObservation)
      .def("get_shot_observations", &map::TracksManager::GetShotObservations)
      .def("get_track_observations", &map::TracksManager::GetTrackObservations)
      .def("construct_sub_tracks_manager",
           &map::TracksManager::ConstructSubTracksManager)
      .def("write_to_file", &map::TracksManager::WriteToFile)
      .def("as_string", &map::TracksManager::AsString)
      .def("get_all_common_observations",
           &map::TracksManager::GetAllCommonObservations,
           py::call_guard<py::gil_scoped_release>())
      .def("get_all_pairs_connectivity",
           &map::TracksManager::GetAllPairsConnectivity,
           py::arg("shots") = std::vector<map::ShotId>(),
           py::arg("tracks") = std::vector<map::TrackId>(),
           py::call_guard<py::gil_scoped_release>());

  py::class_<map::PanoShotView>(m, "PanoShotView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::PanoShotView::NumberOfShots)
      .def(
          "items",
          [](const map::PanoShotView &sv) {
            auto &shots = sv.GetShots();
            return py::make_ref_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](const map::PanoShotView &sv) {
            auto &shots = sv.GetShots();
            return py::make_ref_value_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::PanoShotView &sv) {
            const auto &shots = sv.GetShots();
            return py::make_key_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::PanoShotView &sv) {
            const auto &shots = sv.GetShots();
            return py::make_key_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::PanoShotView::GetShot,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::PanoShotView::GetShot,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::PanoShotView::HasShot);

  py::class_<map::ShotView>(m, "ShotView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::ShotView::NumberOfShots)
      .def(
          "items",
          [](const map::ShotView &sv) {
            const auto &shots = sv.GetShots();
            return py::make_ref_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](const map::ShotView &sv) {
            const auto &shots = sv.GetShots();
            return py::make_ref_value_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::ShotView &sv) {
            const auto &shots = sv.GetShots();
            return py::make_key_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::ShotView &sv) {
            const auto &shots = sv.GetShots();
            return py::make_key_iterator(shots.begin(), shots.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::ShotView::GetShot,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::ShotView::GetShot,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::ShotView::HasShot);

  py::class_<map::LandmarkView>(m, "LandmarkView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::LandmarkView::NumberOfLandmarks)
      .def(
          "items",
          [](const map::LandmarkView &sv) {
            auto &lms = sv.GetLandmarks();
            return py::make_ref_iterator(lms.begin(), lms.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](const map::LandmarkView &sv) {
            auto &lms = sv.GetLandmarks();
            return py::make_ref_value_iterator(lms.begin(), lms.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::LandmarkView &sv) {
            const auto &lms = sv.GetLandmarks();
            return py::make_key_iterator(lms.begin(), lms.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::LandmarkView &sv) {
            const auto &lms = sv.GetLandmarks();
            return py::make_key_iterator(lms.begin(), lms.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::LandmarkView::GetLandmark,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::LandmarkView::GetLandmark,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::LandmarkView::HasLandmark);

  py::class_<map::CameraView>(m, "CameraView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::CameraView::NumberOfCameras)
      .def(
          "items",
          [](const map::CameraView &sv) {
            const auto &cams = sv.GetCameras();
            return py::make_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](map::CameraView &sv) {
            auto &cams = sv.GetCameras();
            return py::make_ref_value_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::CameraView &sv) {
            const auto &cams = sv.GetCameras();
            return py::make_key_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::CameraView &sv) {
            const auto &cams = sv.GetCameras();
            return py::make_key_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::CameraView::GetCamera,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::CameraView::GetCamera,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::CameraView::HasCamera);

  py::class_<map::BiasView>(m, "BiasView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::BiasView::NumberOfBiases)
      .def(
          "items",
          [](const map::BiasView &sv) {
            const auto &biases = sv.GetBiases();
            return py::make_iterator(biases.begin(), biases.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](map::BiasView &sv) {
            auto &biases = sv.GetBiases();
            return py::make_ref_value_iterator(biases.begin(), biases.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::BiasView &sv) {
            const auto &biases = sv.GetBiases();
            return py::make_key_iterator(biases.begin(), biases.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::BiasView &sv) {
            const auto &biases = sv.GetBiases();
            return py::make_key_iterator(biases.begin(), biases.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::BiasView::GetBias,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::BiasView::GetBias,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::BiasView::HasBias);

  py::class_<map::RigCameraView>(m, "RigCameraView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::RigCameraView::NumberOfRigCameras)
      .def(
          "items",
          [](const map::RigCameraView &sv) {
            const auto &cams = sv.GetRigCameras();
            return py::make_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](map::RigCameraView &sv) {
            auto &cams = sv.GetRigCameras();
            return py::make_ref_value_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::RigCameraView &sv) {
            const auto &cams = sv.GetRigCameras();
            return py::make_key_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::RigCameraView &sv) {
            const auto &cams = sv.GetRigCameras();
            return py::make_key_iterator(cams.begin(), cams.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::RigCameraView::GetRigCamera,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::RigCameraView::GetRigCamera,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::RigCameraView::HasRigCamera);

  py::class_<map::RigInstanceView>(m, "RigInstanceView")
      .def(py::init<map::Map &>(),
           py::keep_alive<1, 2>())  // Keep map alive while view is used
      .def("__len__", &map::RigInstanceView::NumberOfRigInstances)
      .def(
          "items",
          [](const map::RigInstanceView &sv) {
            const auto &instances = sv.GetRigInstances();
            return py::make_iterator(instances.begin(), instances.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "values",
          [](map::RigInstanceView &sv) {
            auto &instances = sv.GetRigInstances();
            return py::make_ref_value_iterator(instances.begin(),
                                               instances.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "__iter__",
          [](const map::RigInstanceView &sv) {
            const auto &instances = sv.GetRigInstances();
            return py::make_iterator(instances.begin(), instances.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def(
          "keys",
          [](const map::RigInstanceView &sv) {
            const auto &instances = sv.GetRigInstances();
            return py::make_key_iterator(instances.begin(), instances.end());
          },
          py::keep_alive<0, 1>())  // Keep view alive while iterator is used
      .def("get", &map::RigInstanceView::GetRigInstance,
           py::return_value_policy::reference_internal)
      .def("__getitem__", &map::RigInstanceView::GetRigInstance,
           py::return_value_policy::reference_internal)
      .def("__contains__", &map::RigInstanceView::HasRigInstance);

  mapCls.def(py::init())
      .def_static("deep_copy", &map::Map::DeepCopy,
                  py::return_value_policy::reference_internal,
                  py::call_guard<py::gil_scoped_release>())
      // Camera
      .def("create_camera", &map::Map::CreateCamera, py::arg("camera"),
           py::return_value_policy::reference_internal)
      .def("get_camera",
           py::overload_cast<const map::CameraId &>(&map::Map::GetCamera),
           py::return_value_policy::reference_internal)
      // Bias
      .def("set_bias", &map::Map::SetBias,
           py::return_value_policy::reference_internal)
      .def("get_bias", &map::Map::GetBias,
           py::return_value_policy::reference_internal)
      // Rigs
      .def("create_rig_camera", &map::Map::CreateRigCamera,
           py::return_value_policy::reference_internal)
      .def("create_rig_instance", &map::Map::CreateRigInstance,
           py::return_value_policy::reference_internal)
      .def("update_rig_instance", &map::Map::UpdateRigInstance,
           py::return_value_policy::reference_internal)
      .def("remove_rig_instance", &map::Map::RemoveRigInstance)
      // Landmark
      .def("create_landmark", &map::Map::CreateLandmark, py::arg("lm_id"),
           py::arg("global_position"),
           py::return_value_policy::reference_internal)
      .def("remove_landmark", (void(map::Map::*)(const map::Landmark *const)) &
                                  map::Map::RemoveLandmark)
      .def("remove_landmark", (void(map::Map::*)(const map::LandmarkId &)) &
                                  map::Map::RemoveLandmark)
      .def("has_landmark", &map::Map::HasLandmark)
      .def("get_landmark",
           py::overload_cast<const map::LandmarkId &>(&map::Map::GetLandmark),
           py::return_value_policy::reference_internal)
      .def("clear_observations_and_landmarks",
           &map::Map::ClearObservationsAndLandmarks)
      .def("clean_landmarks_below_min_observations",
           &map::Map::CleanLandmarksBelowMinObservations)
      // Shot
      .def(
          "create_shot",
          py::overload_cast<const map::ShotId &, const map::CameraId &,
                            const map::RigCameraId &,
                            const map::RigInstanceId &, const geometry::Pose &>(
              &map::Map::CreateShot),
          py::return_value_policy::reference_internal)
      .def("create_shot",
           py::overload_cast<const map::ShotId &, const map::CameraId &,
                             const map::RigCameraId &,
                             const map::RigInstanceId &>(&map::Map::CreateShot),
           py::return_value_policy::reference_internal)
      .def("remove_shot", &map::Map::RemoveShot)
      .def("get_shot",
           py::overload_cast<const map::ShotId &>(&map::Map::GetShot),
           py::return_value_policy::reference_internal)
      .def("update_shot", &map::Map::UpdateShot,
           py::return_value_policy::reference_internal)
      // Pano Shot
      .def("create_pano_shot", &map::Map::CreatePanoShot,
           py::return_value_policy::reference_internal)
      .def("remove_pano_shot", &map::Map::RemovePanoShot)
      .def("get_pano_shot",
           py::overload_cast<const map::ShotId &>(&map::Map::GetPanoShot),
           py::return_value_policy::reference_internal)
      .def("update_pano_shot", &map::Map::UpdatePanoShot,
           py::return_value_policy::reference_internal)
      // Observation
      .def("add_observation",
           (void(map::Map::*)(map::Shot *const, map::Landmark *const,
                              const map::Observation &)) &
               map::Map::AddObservation,
           py::arg("shot"), py::arg("landmark"), py::arg("observation"))
      .def("add_observation",
           (void(map::Map::*)(const map::ShotId &, const map::LandmarkId &,
                              const map::Observation &)) &
               map::Map::AddObservation,
           py::arg("shot_Id"), py::arg("landmark_id"), py::arg("observation"))
      .def("remove_observation",
           (void(map::Map::*)(const map::ShotId &, const map::LandmarkId &)) &
               map::Map::RemoveObservation,
           py::arg("shot"), py::arg("landmark"))
      // Getters
      .def("get_shots", &map::Map::GetShotView)
      .def("get_pano_shots", &map::Map::GetPanoShotView)
      .def("get_cameras", &map::Map::GetCameraView)
      .def("get_biases", &map::Map::GetBiasView)
      .def("get_camera_view", &map::Map::GetCameraView)
      .def("get_landmarks", &map::Map::GetLandmarkView)
      .def("get_landmark_view", &map::Map::GetLandmarkView)
      .def("set_reference", &map::Map::SetTopocentricConverter)
      // Reference
      .def("get_reference",
           [](const map::Map &map) {
             py::module::import("opensfm.pygeo");
             return map.GetTopocentricConverter();
           })
      // Tracks manager x Reconstruction intersection
      .def("compute_reprojection_errors", &map::Map::ComputeReprojectionErrors)
      .def("get_valid_observations", &map::Map::GetValidObservations)
      .def("to_tracks_manager", &map::Map::ToTracksManager);
}
