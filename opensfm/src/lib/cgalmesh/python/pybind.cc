#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

#include "cgalmesh/delaunay_mesher.h"

namespace py = pybind11;

namespace {

// Thin pybind wrapper holding one triangulation across the centroid -> classify
// -> extract round-trip.
class DelaunayMesherPy {
 public:
  explicit DelaunayMesherPy(
      const py::array_t<double, py::array::c_style | py::array::forcecast>&
          points) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
      throw std::invalid_argument("points must be an (N, 3) array");
    }
    const std::size_t n = static_cast<std::size_t>(points.shape(0));
    py::gil_scoped_release release;
    mesher_ = std::make_unique<cgalmesh::DelaunayMesher>(points.data(), n);
  }

  int num_cells() const { return mesher_->num_cells(); }

  // (Nc, 3) float64 finite-cell centroids; row i is the cell labelled i.
  py::array_t<double> cell_centroids() const {
    const std::vector<double>& c = mesher_->centroids();
    const std::size_t nc = c.size() / 3;
    py::array_t<double> out({nc, std::size_t(3)});
    std::copy(c.begin(), c.end(), out.mutable_data());
    return out;
  }

  // Hybrid graph cut → per finite cell label (Nc,) int8 (1 = inside).
  // origins/targets are (R, 3) camera centres and observed surface points;
  // cell_sign is the (Nc,) int8 sign prior (1 = inside) or empty to disable it.
  py::array_t<int8_t> graphcut_labels(
      const py::array_t<double, py::array::c_style | py::array::forcecast>&
          origins,
      const py::array_t<double, py::array::c_style | py::array::forcecast>&
          targets,
      const py::array_t<int8_t, py::array::c_style | py::array::forcecast>&
          cell_sign,
      double alpha_sign, double alpha_vis, double lambda_qual,
      double back_eps) const {
    if (origins.ndim() != 2 || origins.shape(1) != 3 || targets.ndim() != 2 ||
        targets.shape(1) != 3 || origins.shape(0) != targets.shape(0)) {
      throw std::invalid_argument(
          "origins and targets must be matching (R, 3) arrays");
    }
    const bool has_sign = cell_sign.size() > 0;
    if (has_sign && cell_sign.size() != num_cells()) {
      throw std::invalid_argument(
          "cell_sign length must equal num_cells() (or be empty)");
    }
    const std::size_t n_rays = static_cast<std::size_t>(origins.shape(0));
    std::vector<int8_t> labels;
    {
      py::gil_scoped_release release;
      labels = mesher_->graphcut_labels(
          origins.data(), targets.data(), n_rays,
          has_sign ? cell_sign.data() : nullptr, alpha_sign, alpha_vis,
          lambda_qual, back_eps);
    }
    py::array_t<int8_t> out(static_cast<py::ssize_t>(labels.size()));
    std::copy(labels.begin(), labels.end(), out.mutable_data());
    return out;
  }

  // labels: (Nc,) int8, 1 = inside, 0 = outside.  Returns (verts (Nv,3) float64,
  // faces (Nf,3) int32).
  py::tuple extract_surface(
      const py::array_t<int8_t, py::array::c_style | py::array::forcecast>&
          labels,
      bool drop_hull, double max_edge, double min_quality,
      int min_component_faces, int max_hole_edges,
      bool remove_self_intersections) const {
    std::vector<double> verts;
    std::vector<int> faces;
    {
      py::gil_scoped_release release;
      mesher_->extract_surface(labels.data(),
                               static_cast<std::size_t>(labels.size()),
                               drop_hull, max_edge, min_quality,
                               min_component_faces, max_hole_edges,
                               remove_self_intersections, &verts, &faces);
    }
    const std::size_t nv = verts.size() / 3;
    const std::size_t nf = faces.size() / 3;
    py::array_t<double> verts_arr({nv, std::size_t(3)});
    std::copy(verts.begin(), verts.end(), verts_arr.mutable_data());
    py::array_t<int> faces_arr({nf, std::size_t(3)});
    std::copy(faces.begin(), faces.end(), faces_arr.mutable_data());
    return py::make_tuple(verts_arr, faces_arr);
  }

 private:
  std::unique_ptr<cgalmesh::DelaunayMesher> mesher_;
};

}  // namespace

PYBIND11_MODULE(pycgalmesh, m) {
  m.doc() = "CGAL Delaunay-tetrahedralisation surface meshing.";
  py::class_<DelaunayMesherPy>(m, "DelaunayMesher")
      .def(py::init<const py::array_t<double, py::array::c_style |
                                                  py::array::forcecast>&>(),
           py::arg("points"))
      .def_property_readonly("num_cells", &DelaunayMesherPy::num_cells)
      .def("cell_centroids", &DelaunayMesherPy::cell_centroids)
      .def("graphcut_labels", &DelaunayMesherPy::graphcut_labels,
           py::arg("origins"), py::arg("targets"), py::arg("cell_sign"),
           py::arg("alpha_sign") = 3.0, py::arg("alpha_vis") = 1.0,
           py::arg("lambda_qual") = 0.1, py::arg("back_eps") = 0.0)
      .def("extract_surface", &DelaunayMesherPy::extract_surface,
           py::arg("labels"), py::arg("drop_hull") = false,
           py::arg("max_edge") = 0.0, py::arg("min_quality") = 0.0,
           py::arg("min_component_faces") = 0, py::arg("max_hole_edges") = 0,
           py::arg("remove_self_intersections") = false);
}
