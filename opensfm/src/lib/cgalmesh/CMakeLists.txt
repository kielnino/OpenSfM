# CGAL Delaunay-tetrahedralisation meshing (pycgalmesh).
#
# Isolated in its own translation unit / pybind module so the (slow) CGAL header
# compiles do not inflate rebuilds of pydense.  Header-only CGAL: the CGAL::CGAL
# target carries the include dirs and the gmp/mpfr links.

pybind11_add_module(pycgalmesh python/pybind.cc)
target_include_directories(pycgalmesh PRIVATE ${opensfm_SOURCE_DIR})
target_link_libraries(pycgalmesh PRIVATE CGAL::CGAL)

# CGAL + numerical predicates want C++17 and disabled fast-math reordering.
target_compile_features(pycgalmesh PRIVATE cxx_std_17)

set_target_properties(pycgalmesh PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/..")
install(TARGETS pycgalmesh LIBRARY DESTINATION .)
