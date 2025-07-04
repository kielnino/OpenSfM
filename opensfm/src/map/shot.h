#pragma once
#include <foundation/optional.h>
#include <geometry/camera.h>
#include <geometry/pose.h>
#include <map/defines.h>
#include <map/landmark.h>
#include <map/observation.h>
#include <map/rig.h>

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>

namespace map {
class Map;

struct ShotMesh {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void SetVertices(const MatXd& vertices) { vertices_ = vertices; }
  void SetFaces(const MatXd& faces) { faces_ = faces; }
  MatXd GetFaces() const { return faces_; }
  MatXd GetVertices() const { return vertices_; }
  MatXd vertices_;
  MatXd faces_;
};

struct ShotMeasurements {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  foundation::OptionalValue<double> capture_time_;
  foundation::OptionalValue<Vec3d> gps_position_;
  foundation::OptionalValue<double> gps_accuracy_;
  foundation::OptionalValue<double> compass_accuracy_;
  foundation::OptionalValue<double> compass_angle_;
  foundation::OptionalValue<Vec3d> gravity_down_;
  foundation::OptionalValue<double> opk_accuracy_;
  foundation::OptionalValue<Vec3d> opk_angles_;
  foundation::OptionalValue<int> orientation_;
  foundation::OptionalValue<std::string> sequence_key_;
  void Set(const ShotMeasurements& other);

  // Store any additional attributes
  std::map<std::string, std::string> attributes_;
  const auto& GetAttributes() const { return attributes_; }
  auto& GetMutableAttributes() { return attributes_; }
  void SetAttributes(const std::map<std::string, std::string>& attributes) {
    attributes_ = attributes;
  }
};

class Shot {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Shot construction
  Shot(const ShotId& shot_id, const geometry::Camera* const shot_camera,
       RigInstance* rig_instance, RigCamera* rig_camera,
       const geometry::Pose& pose);
  Shot(const ShotId& shot_id, const geometry::Camera* const shot_camera,
       RigInstance* rig_instance, RigCamera* rig_camera);
  Shot(const ShotId& shot_id, const geometry::Camera& shot_camera,
       const geometry::Pose& pose);
  ShotId GetId() const { return id_; }

  // Rig
  bool IsInRig() const;
  void SetRig(RigInstance* rig_instance, RigCamera* rig_camera);
  RigInstance* GetRigInstance() const { return rig_instance_; }
  const RigCamera* GetRigCamera() const { return rig_camera_; }
  const RigInstanceId& GetRigInstanceId() const;
  const RigCameraId& GetRigCameraId() const;

  // Pose
  void SetPose(const geometry::Pose& pose);
  const geometry::Pose* GetPose() const;
  geometry::Pose* GetPose();
  Mat4d GetWorldToCam() const { return GetPose()->WorldToCamera(); }
  Mat4d GetCamToWorld() const { return GetPose()->CameraToWorld(); }

  // Landmark management
  const std::map<
      Landmark*, Observation, KeyCompare,
      Eigen::aligned_allocator<std::pair<Landmark* const, Observation>>>&
  GetLandmarkObservations() const {
    return landmark_observations_;
  }
  std::map<Landmark*, Observation, KeyCompare,
           Eigen::aligned_allocator<std::pair<Landmark* const, Observation>>>&
  GetLandmarkObservations() {
    return landmark_observations_;
  }
  std::vector<Landmark*> ComputeValidLandmarks() {
    std::vector<Landmark*> valid_landmarks;
    valid_landmarks.reserve(landmark_observations_.size());
    for (const auto& lm_obs : landmark_observations_) {
      valid_landmarks.push_back(lm_obs.first);
    }
    return valid_landmarks;
  }

  // Observation management
  const Observation& GetObservation(const FeatureId id) const {
    return landmark_observations_.at(landmark_id_.at(id));
  }
  void CreateObservation(Landmark* lm, const Observation& obs) {
    landmark_observations_.insert(std::make_pair(lm, obs));
    landmark_id_.insert(std::make_pair(obs.feature_id, lm));
  }
  Observation* GetLandmarkObservation(Landmark* lm) {
    return &landmark_observations_.at(lm);
  }
  Landmark* GetObservationLandmark(const FeatureId id) {
    auto find_landmark = landmark_id_.find(id);
    if (find_landmark == landmark_id_.end()) {
      return nullptr;
    } else {
      return find_landmark->second;
    }
  }
  void RemoveLandmarkObservation(const FeatureId id);

  // Metadata such as GPS, IMU, time
  const ShotMeasurements& GetShotMeasurements() const {
    return shot_measurements_;
  }
  ShotMeasurements& GetShotMeasurements() { return shot_measurements_; }
  void SetShotMeasurements(const ShotMeasurements& other) {
    shot_measurements_.Set(other);
  }

  // Comparisons
  bool operator==(const Shot& shot) const { return id_ == shot.id_; }
  bool operator!=(const Shot& shot) const { return !(*this == shot); }
  bool operator<(const Shot& shot) const { return id_ < shot.id_; }
  bool operator<=(const Shot& shot) const { return id_ <= shot.id_; }
  bool operator>(const Shot& shot) const { return id_ > shot.id_; }
  bool operator>=(const Shot& shot) const { return id_ >= shot.id_; }
  const geometry::Camera* GetCamera() const { return shot_camera_; }

  // Camera-related
  Vec2d Project(const Vec3d& global_pos) const;
  MatX2d ProjectMany(const MatX3d& points) const;
  Vec3d Bearing(const Vec2d& point) const;
  MatX3d BearingMany(const MatX2d& points) const;

  // Covariance
  MatXd GetCovariance() const { return covariance_.Value(); }
  void SetCovariance(const MatXd& cov) { covariance_.SetValue(cov); }

 public:
  const ShotId id_;  // the file name

  // Ad-hoc merge-specific data
  ShotMesh mesh;
  long int merge_cc{0};
  double scale{1.0};

 private:
  geometry::Pose GetPoseInRig() const;

  // Pose
  mutable std::unique_ptr<geometry::Pose> pose_;
  foundation::OptionalValue<MatXd> covariance_;

  // Rig data (can optionally belong to the shot)
  foundation::OptionalValue<RigInstance> own_rig_instance_;
  foundation::OptionalValue<RigCamera> own_rig_camera_;
  RigInstance* rig_instance_;
  RigCamera* rig_camera_;

  // Camera pointer (can optionally belong to the shot)
  foundation::OptionalValue<geometry::Camera> own_camera_;
  const geometry::Camera* const shot_camera_;

  // Metadata
  ShotMeasurements shot_measurements_;

  // In OpenSfM, we use a map to reproduce a similar behaviour
  std::map<Landmark*, Observation, KeyCompare,
           Eigen::aligned_allocator<std::pair<Landmark* const, Observation>>>
      landmark_observations_;
  std::unordered_map<FeatureId, Landmark*> landmark_id_;
};
}  // namespace map
