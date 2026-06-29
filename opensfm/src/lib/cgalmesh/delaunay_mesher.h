#pragma once

// Delaunay-tetrahedralisation surface meshing.
//
// Given a surface point cloud, build a 3-D Delaunay tetrahedralisation, let the
// caller label each tetrahedron INSIDE/OUTSIDE (e.g. by the TSDF sign at the
// cell centroid, with an oriented-point fallback in empty space), and extract
// the triangles that sit on the INSIDE<->OUTSIDE boundary as a watertight,
// manifold mesh.  The Delaunay connectivity bridges gaps in the cloud, so the
// holes that a dual-contouring mesh leaves open get filled by interpolation.
//
// Usage (one triangulation, two calls):
//   m = DelaunayMesher(points)            # builds + indexes finite cells
//   centroids = m.cell_centroids()        # (Nc, 3); cell i == row i
//   labels = classify(centroids)          # caller's oracle, (Nc,) int8
//   verts, faces = m.extract_surface(labels, ...)
//
// Cell indices are assigned once at construction (cell->info()), so
// cell_centroids() and extract_surface() agree regardless of call order.

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cgalmesh {

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
// Vertex info = original input-point index; cell info = sequential finite-cell
// index (the row in the labels / centroid arrays).
using Vb = CGAL::Triangulation_vertex_base_with_info_3<int, K>;
using Cb = CGAL::Triangulation_cell_base_with_info_3<int, K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
using Delaunay = CGAL::Delaunay_triangulation_3<K, Tds>;
using Point = K::Point_3;
using Vector = K::Vector_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;

namespace PMP = CGAL::Polygon_mesh_processing;

// Compact forward-star max-flow (Dinic) for the min-cut: one node per finite
// tetra + a source ("outside") and sink ("inside").  A vector-of-vectors graph
// (boost::adjacency_list) costs several GB and is slow to build at this scale;
// here each arc is just (to:int32, next:int32, cap:float) = 12 bytes, paired so
// arc e and e^1 are reverses.  Recursion depth is bounded by the s–t
// level-graph height (≈ the chunk's cell diameter), so the recursive DFS is
// safe.
class MaxFlow {
 public:
  explicit MaxFlow(int n_nodes) : n_(n_nodes), head_(n_nodes, -1) {}

  void reserve(std::size_t n_arcs) {
    to_.reserve(2 * n_arcs);
    next_.reserve(2 * n_arcs);
    cap_.reserve(2 * n_arcs);
  }

  // Add an undirected/directed pair: arc u->v (cap_uv) and v->u (cap_vu).
  void add_edge(int u, int v, float cap_uv, float cap_vu) {
    to_.push_back(v);
    cap_.push_back(cap_uv);
    next_.push_back(head_[u]);
    head_[u] = static_cast<int>(to_.size()) - 1;
    to_.push_back(u);
    cap_.push_back(cap_vu);
    next_.push_back(head_[v]);
    head_[v] = static_cast<int>(to_.size()) - 1;
  }

  // FIFO push-relabel with periodic global relabeling.  Dinic augments one s-t
  // path per DFS, which is hopeless when millions of unit-weight visibility rays
  // create millions of augmenting paths; push-relabel moves flow in bulk via
  // node excess/heights and converges in near-linear time on graphs this size.
  void max_flow(int s, int t) {
    s_ = s;
    t_ = t;
    const int N = n_;
    height_.assign(N, 0);
    excess_.assign(N, 0.0);
    cur_.assign(N, -1);

    global_relabel();  // exact residual distance-to-t; height_[s] := N
    // Initial preflow: saturate every arc out of the source.
    for (int e = head_[s_]; e != -1; e = next_[e]) {
      const float c = cap_[e];
      if (c > kEps) {
        cap_[e] = 0.0f;
        cap_[e ^ 1] += c;
        excess_[to_[e]] += c;
      }
    }

    std::queue<int> active;
    std::vector<char> inq(N, 0);
    auto enqueue = [&](int v) {
      if (v != s_ && v != t_ && !inq[v] && excess_[v] > kEps) {
        active.push(v);
        inq[v] = 1;
      }
    };
    for (int i = 0; i < N; ++i) enqueue(i);

    long relabels = 0;
    const long gr_period = static_cast<long>(N) + 1;  // global relabel cadence
    while (!active.empty()) {
      const int u = active.front();
      active.pop();
      inq[u] = 0;
      while (excess_[u] > kEps) {
        const int e = cur_[u];
        if (e == -1) {  // scanned all arcs → relabel
          int mh = 2 * N;
          for (int a = head_[u]; a != -1; a = next_[a]) {
            if (cap_[a] > kEps && height_[to_[a]] + 1 < mh) {
              mh = height_[to_[a]] + 1;
            }
          }
          height_[u] = mh;
          cur_[u] = head_[u];
          ++relabels;
          if (height_[u] >= 2 * N) break;  // disconnected from t — excess trapped
          continue;
        }
        const int v = to_[e];
        if (cap_[e] > kEps && height_[u] == height_[v] + 1) {
          const double d = std::min<double>(excess_[u], cap_[e]);
          cap_[e] -= static_cast<float>(d);
          cap_[e ^ 1] += static_cast<float>(d);
          excess_[u] -= d;
          excess_[v] += d;
          enqueue(v);
        } else {
          cur_[u] = next_[e];
        }
      }
      if (relabels >= gr_period) {  // refresh exact heights, rebuild active set
        global_relabel();
        relabels = 0;
        std::queue<int> empty;
        std::swap(active, empty);
        std::fill(inq.begin(), inq.end(), 0);
        for (int i = 0; i < N; ++i)
          if (height_[i] < 2 * N) enqueue(i);
      }
    }
  }

  // After max_flow: nodes that can still reach the SINK in the residual graph
  // form the sink ("inside") side of the min cut.  Reverse-BFS from the sink is
  // the robust choice — it is a valid min-cut side even for the max *preflow*
  // push-relabel leaves (trapped excess on nodes disconnected from the sink),
  // whereas "reachable from the source" would mislabel those trapped cells.
  std::vector<char> sink_side() const {
    std::vector<char> reach(n_, 0);
    std::queue<int> q;
    reach[t_] = 1;
    q.push(t_);
    while (!q.empty()) {
      const int v = q.front();
      q.pop();
      for (int f = head_[v]; f != -1; f = next_[f]) {
        // Residual arc u->v exists iff cap_[f^1] > 0 (f is v's out-arc v->u).
        const int u = to_[f];
        if (!reach[u] && cap_[f ^ 1] > kEps) {
          reach[u] = 1;
          q.push(u);
        }
      }
    }
    return reach;
  }

 private:
  static constexpr float kEps = 1e-12f;

  // Exact heights = residual graph distance to the sink (reverse BFS).  The
  // dominant push-relabel speedup; run once at the start and periodically.
  void global_relabel() {
    const int N = n_;
    std::fill(height_.begin(), height_.end(), 2 * N);
    std::queue<int> q;
    height_[t_] = 0;
    q.push(t_);
    while (!q.empty()) {
      const int v = q.front();
      q.pop();
      for (int f = head_[v]; f != -1; f = next_[f]) {
        // Out-arc f is v->to_[f]; its reverse f^1 is the arc to_[f]->v whose
        // residual capacity gates flow from to_[f] toward the sink through v.
        const int u = to_[f];
        if (height_[u] >= 2 * N && cap_[f ^ 1] > kEps) {
          height_[u] = height_[v] + 1;
          q.push(u);
        }
      }
    }
    height_[s_] = N;
    std::copy(head_.begin(), head_.end(), cur_.begin());
  }

  int n_, s_ = 0, t_ = 0;
  std::vector<int> head_, to_, next_, cur_;
  std::vector<float> cap_;
  std::vector<int> height_;
  std::vector<double> excess_;
};

class DelaunayMesher {
 public:
  // Build the Delaunay tetrahedralisation from |pts| (n x 3, row-major, world
  // coordinates).  Finite cells are indexed 0..num_cells()-1 in iteration order
  // and their centroids cached.
  DelaunayMesher(const double* pts, std::size_t n) {
    if (n < 4) {
      throw std::invalid_argument(
          "DelaunayMesher: need at least 4 points for a tetrahedralisation");
    }
    std::vector<std::pair<Point, int>> pts_with_info;
    pts_with_info.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      pts_with_info.emplace_back(
          Point(pts[3 * i + 0], pts[3 * i + 1], pts[3 * i + 2]),
          static_cast<int>(i));
    }
    dt_.insert(pts_with_info.begin(), pts_with_info.end());

    // Index finite cells and cache centroids.
    int idx = 0;
    for (auto c = dt_.finite_cells_begin(); c != dt_.finite_cells_end(); ++c) {
      c->info() = idx++;
      const Point& a = c->vertex(0)->point();
      const Point& b = c->vertex(1)->point();
      const Point& d = c->vertex(2)->point();
      const Point& e = c->vertex(3)->point();
      centroids_.push_back(0.25 * (a.x() + b.x() + d.x() + e.x()));
      centroids_.push_back(0.25 * (a.y() + b.y() + d.y() + e.y()));
      centroids_.push_back(0.25 * (a.z() + b.z() + d.z() + e.z()));
    }
    num_cells_ = idx;
  }

  int num_cells() const { return num_cells_; }

  // Flat (Nc*3) centroid buffer, row i == finite cell with info() == i.
  const std::vector<double>& centroids() const { return centroids_; }

  // Extract the INSIDE<->OUTSIDE boundary as a manifold triangle mesh.
  //   labels        : per finite cell, 1 = inside (solid), 0 = outside (free).
  //   drop_hull      : if true, skip facets against the infinite cell (the
  //   outer
  //                    convex-hull cap) — leaves an open surface instead of a
  //                    closed solid.  Interior holes are still filled.
  //   max_edge       : if > 0, drop boundary facets whose longest edge exceeds
  //                    this (world units) — removes the giant triangles that span
  //                    unseen free space / the convex-hull skirt while keeping
  //                    the fine surface and reasonable hole bridges.
  //   min_quality    : if > 0, drop sliver / "spider-web" facets — those whose
  //                    shortest altitude is below this fraction of their longest
  //                    edge (2*area / longest_edge^2 < min_quality).  ~0.05.
  //   min_component_faces : if > 0, drop connected components with fewer faces
  //                    (removes the scattered speckle triangles).
  //   max_hole_edges : if > 0, fill boundary holes whose cycle has at most this
  //                    many edges (the tiny holes; the scene's outer boundary,
  //                    a long cycle, is left open).
  //   remove_self_intersect : if true, a final CGAL pass removes self-
  //                    intersecting facets (folds/overlaps).
  // Returns the triangle soup, repaired + oriented into a manifold:
  //   out_verts : flat (Nv*3) double positions.
  //   out_faces : flat (Nf*3) int vertex indices (0-based into out_verts).
  void extract_surface(const int8_t* labels, std::size_t n_labels,
                       bool drop_hull, double max_edge, double min_quality,
                       int min_component_faces, int max_hole_edges,
                       bool remove_self_intersect,
                       std::vector<double>* out_verts,
                       std::vector<int>* out_faces) const {
    if (static_cast<int>(n_labels) != num_cells_) {
      throw std::invalid_argument(
          "extract_surface: labels length must equal num_cells()");
    }
    const double max_edge_sq = max_edge * max_edge;

    // Soup of the boundary facets.  Points are the subset of input vertices on
    // the surface; we dedup by input index so shared vertices are not torn.
    std::vector<Point> soup_pts;
    std::vector<std::vector<std::size_t>> soup_faces;
    // Map input vertex index -> soup index lazily via a hash.
    std::unordered_map<int, std::size_t> used;
    used.reserve(num_cells_);

    auto soup_index = [&](Delaunay::Vertex_handle v) -> std::size_t {
      const int src = v->info();
      auto it = used.find(src);
      if (it != used.end()) {
        return it->second;
      }
      const std::size_t s = soup_pts.size();
      soup_pts.push_back(v->point());
      used.emplace(src, s);
      return s;
    };

    for (auto c = dt_.finite_cells_begin(); c != dt_.finite_cells_end(); ++c) {
      if (labels[c->info()] != 1) {
        continue;  // only walk inside cells
      }
      for (int i = 0; i < 4; ++i) {
        const Delaunay::Cell_handle nb = c->neighbor(i);
        const bool nb_inf = dt_.is_infinite(nb);
        if (nb_inf && drop_hull) {
          continue;
        }
        const int nb_label = nb_inf ? 0 : labels[nb->info()];
        if (nb_label == 1) {
          continue;  // both inside: not a boundary facet
        }

        // Facet (c, i): the three vertices other than vertex i.  Wind so the
        // normal points away from the apex (vertex i, which is inside) -> the
        // mesh is outward-oriented by construction.
        const Point& apex = c->vertex(i)->point();
        Delaunay::Vertex_handle vh[3];
        int k = 0;
        for (int j = 0; j < 4; ++j) {
          if (j != i) {
            vh[k++] = c->vertex(j);
          }
        }
        const Point& p0 = vh[0]->point();
        const Point& p1 = vh[1]->point();
        const Point& p2 = vh[2]->point();
        const double max_e_sq = std::max(
            {(p1 - p0).squared_length(), (p2 - p1).squared_length(),
             (p0 - p2).squared_length()});
        if (max_edge_sq > 0.0 && max_e_sq > max_edge_sq) {
          continue;  // giant triangle spanning unseen / free space
        }
        const Vector nrm = CGAL::cross_product(p1 - p0, p2 - p0);
        if (min_quality > 0.0) {
          // |nrm| == 2*area; shortest altitude = 2*area / longest_edge.
          const double two_area = std::sqrt(nrm.squared_length());
          if (two_area < min_quality * max_e_sq) {
            continue;  // sliver / spider-web facet
          }
        }
        // Centroid of the facet, vector apex->centroid points outward.
        const Vector out_dir((p0.x() + p1.x() + p2.x()) / 3.0 - apex.x(),
                             (p0.y() + p1.y() + p2.y()) / 3.0 - apex.y(),
                             (p0.z() + p1.z() + p2.z()) / 3.0 - apex.z());
        const bool flip = (nrm * out_dir) < 0.0;
        const std::size_t s0 = soup_index(vh[0]);
        const std::size_t s1 = soup_index(vh[flip ? 2 : 1]);
        const std::size_t s2 = soup_index(vh[flip ? 1 : 2]);
        soup_faces.push_back({s0, s1, s2});
      }
    }

    // Repair + orient the soup into a consistent manifold (duplicates
    // non-manifold vertices, drops degenerate/duplicate triangles).
    PMP::repair_polygon_soup(soup_pts, soup_faces);
    PMP::orient_polygon_soup(soup_pts, soup_faces);
    SurfaceMesh mesh;
    PMP::polygon_soup_to_polygon_mesh(soup_pts, soup_faces, mesh);
    mesh.collect_garbage();

    // Drop scattered speckle components (and any debris the filters detached).
    if (min_component_faces > 0) {
      PMP::keep_large_connected_components(
          mesh, static_cast<std::size_t>(min_component_faces));
      mesh.collect_garbage();
    }

    // Fill the tiny holes (those whose border cycle is short); the scene's
    // outer boundary is a long cycle and is left open.  Plain triangulation
    // adds faces only (no new vertices), so per-vertex colour transfer in the
    // caller stays exact.
    if (max_hole_edges > 0) {
      using halfedge_descriptor =
          boost::graph_traits<SurfaceMesh>::halfedge_descriptor;
      std::vector<halfedge_descriptor> borders;
      PMP::extract_boundary_cycles(mesh, std::back_inserter(borders));
      for (const halfedge_descriptor h : borders) {
        std::size_t len = 0;
        halfedge_descriptor hh = h;
        do {
          ++len;
          hh = mesh.next(hh);
        } while (hh != h);
        if (len <= static_cast<std::size_t>(max_hole_edges)) {
          PMP::triangulate_hole(mesh, h);
        }
      }
      mesh.collect_garbage();
    }

    // Final topological cleanup: drop self-intersecting facets (folds the noisy
    // cut or the hole fills can leave) so no triangle covers another.
    if (remove_self_intersect) {
      PMP::experimental::remove_self_intersections(mesh);
      mesh.collect_garbage();
    }

    // Flatten to verts/faces.  Surface_mesh vertex ids are contiguous after
    // collect_garbage(), so v.idx() is a dense 0..Nv-1 index.
    out_verts->clear();
    out_faces->clear();
    out_verts->reserve(mesh.number_of_vertices() * 3);
    for (auto v : mesh.vertices()) {
      const Point& p = mesh.point(v);
      out_verts->push_back(CGAL::to_double(p.x()));
      out_verts->push_back(CGAL::to_double(p.y()));
      out_verts->push_back(CGAL::to_double(p.z()));
    }
    out_faces->reserve(mesh.number_of_faces() * 3);
    for (auto f : mesh.faces()) {
      for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
        out_faces->push_back(static_cast<int>(v.idx()));
      }
    }
  }

  // Label tetrahedra inside/outside by a hybrid Labatut-Pons-Keriven (ICCV 2007)
  // graph cut.  Two data terms feed the min cut: (1) a per-cell SIGN prior
  // (``cell_sign``, e.g. the TSDF / oriented-point oracle: 1 = inside) weighted
  // by ``alpha_sign`` — this anchors the complete zero-set surface; and (2) a
  // VISIBILITY term from each (camera, surface point) ray, weighted by
  // ``alpha_vis`` — a free-space (OUTSIDE) link just in front of the point and a
  // matter (INSIDE) link just behind it, which carves the seen free space the
  // sign oracle wrongly fills.  A constant smoothness ``lambda_qual`` on every
  // interior facet (standing in for the paper's photo-consistency term, since we
  // mesh points not images) plus a convex-hull OUTSIDE prior regularise it.  The
  // global min cut yields a single, complete, carved partition.
  //
  //   origins/targets : R camera centres and observed surface points (world).
  //   cell_sign       : per finite cell, 1 = inside / 0 = outside (or null).
  //   alpha_sign      : weight of the sign prior (0 disables it).
  //   alpha_vis       : per-ray visibility weight.
  //   lambda_qual     : base smoothness (n-link) weight per interior facet.
  //   back_eps        : how far in front of / behind the point (world units) the
  //                     free / matter cells are sampled.
  // Returns a label per finite cell (1 = inside) in cell->info() order.
  std::vector<int8_t> graphcut_labels(const double* origins,
                                      const double* targets, std::size_t n_rays,
                                      const int8_t* cell_sign, double alpha_sign,
                                      double alpha_vis, double lambda_qual,
                                      double back_eps) const {
    const int n = num_cells_;
    std::vector<double> t_source(n, 0.0);  // pull toward OUTSIDE (free space)
    std::vector<double> t_sink(n, 0.0);    // pull toward INSIDE (matter)
    std::unordered_map<long long, double> nlink_vis;  // extra vis on facets

    // (1) Sign data term: anchors the complete surface at the TSDF zero-set.
    if (cell_sign != nullptr && alpha_sign > 0.0) {
      for (int i = 0; i < n; ++i) {
        if (cell_sign[i] == 1) {
          t_sink[i] += alpha_sign;
        } else {
          t_source[i] += alpha_sign;
        }
      }
    }

    auto facet_key = [n](int a, int b) -> long long {
      if (a > b) {
        std::swap(a, b);
      }
      return static_cast<long long>(a) * (n + 1) + b;
    };

    // --- Accumulate visibility evidence from each line of sight. ---
    // The fused TSDF zero-set is a thin sheet whose Delaunay is full of sliver
    // tetrahedra; a parametric segment walk dies on them after ~1 facet (it
    // never reaches free space, so nothing gets carved).  Instead use robust
    // CGAL point location: a free-space (source / OUTSIDE) link just IN FRONT of
    // each observed surface point (toward the camera) and a matter (sink /
    // INSIDE) link just BEHIND it.  The smoothness prior plus the convex-hull
    // prior then propagate the carve up the entire free-space column above the
    // surface and the solid below it — the min cut lands on the surface.
    Delaunay::Cell_handle hint;
    for (std::size_t r = 0; r < n_rays; ++r) {
      const Point tgt(targets[3 * r], targets[3 * r + 1], targets[3 * r + 2]);
      const Point org(origins[3 * r], origins[3 * r + 1], origins[3 * r + 2]);
      Vector to_cam = org - tgt;
      const double cam_len = std::sqrt(to_cam.squared_length());
      if (cam_len < 1e-12) {
        continue;
      }
      to_cam = to_cam / cam_len;  // unit direction target -> camera

      if (r > 0 && r % 1000000 == 0) {
        std::cerr << "graphcut_labels: ray " << r << " / " << n_rays
                  << std::endl;
      }

      // Matter just behind the surface (away from camera) -> INSIDE (sink).
      hint = dt_.locate(tgt - back_eps * to_cam, hint);
      if (!dt_.is_infinite(hint)) {
        t_sink[hint->info()] += alpha_vis;
      }
      // Free space just in front of the surface (toward camera) -> OUTSIDE
      // (source).  Skipped when it lands outside the hull (point on the hull).
      hint = dt_.locate(tgt + back_eps * to_cam, hint);
      if (!dt_.is_infinite(hint)) {
        t_source[hint->info()] += alpha_vis;
      }
    }

    // --- Build the flow graph: nodes 0..n-1 = cells, n = source, n+1 = sink.
    // ---
    const int kSource = n;
    const int kSink = n + 1;
    MaxFlow mf(n + 2);
    mf.reserve(2 * static_cast<std::size_t>(n) + nlink_vis.size());

    for (auto c = dt_.finite_cells_begin(); c != dt_.finite_cells_end(); ++c) {
      const int ci = c->info();
      for (int i = 0; i < 4; ++i) {
        Delaunay::Cell_handle nb = c->neighbor(i);
        if (dt_.is_infinite(nb)) {
          // Hull facet: bias the boundary cell toward OUTSIDE.
          t_source[ci] += lambda_qual;
        } else {
          const int ni = nb->info();
          if (ci < ni) {  // each interior facet once
            double w = lambda_qual;
            auto it = nlink_vis.find(facet_key(ci, ni));
            if (it != nlink_vis.end()) {
              w += it->second;
            }
            mf.add_edge(ci, ni, static_cast<float>(w), static_cast<float>(w));
          }
        }
      }
    }
    for (int i = 0; i < n; ++i) {
      if (t_source[i] > 0.0) {
        mf.add_edge(kSource, i, static_cast<float>(t_source[i]), 0.0f);
      }
      if (t_sink[i] > 0.0) {
        mf.add_edge(i, kSink, static_cast<float>(t_sink[i]), 0.0f);
      }
    }

    std::cerr << "graphcut_labels: max-flow on " << n << " cells, "
              << nlink_vis.size() << " interior facets ..." << std::endl;
    mf.max_flow(kSource, kSink);

    // Cells that can still reach the sink in the residual graph are INSIDE; the
    // rest (cut off from the sink) are carved OUTSIDE.
    const std::vector<char> inside = mf.sink_side();
    std::vector<int8_t> labels(n, 0);
    for (int i = 0; i < n; ++i) {
      labels[i] = inside[i] ? 1 : 0;
    }
    return labels;
  }

 private:
  Delaunay dt_;
  std::vector<double> centroids_;
  int num_cells_ = 0;
};

}  // namespace cgalmesh
