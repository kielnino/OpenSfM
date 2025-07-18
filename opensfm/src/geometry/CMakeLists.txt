# Geometry
set(GEOMETRY_FILES
    essential.h
    absolute_pose.h
    covariance.h
    pose.h
    similarity.h
    camera.h
    functions.h
    camera_distortions_functions.h
    camera_projections_functions.h
    relative_pose.h
    triangulation.h
    src/camera.cc
    src/essential.cc
    src/covariance.cc
    src/triangulation.cc
    src/absolute_pose.cc
    src/relative_pose.cc
    )
add_library(geometry ${GEOMETRY_FILES})
target_link_libraries(geometry
    PRIVATE
    foundation
    ${CERES_LIBRARIES}
    )
target_include_directories(geometry PUBLIC ${CMAKE_SOURCE_DIR})

if (OPENSFM_BUILD_TESTS)
    set(GEOMETRY_TEST_FILES
        test/camera_test.cc
        test/camera_functions_test.cc
        test/covariance_test.cc
        test/point_test.cc
        test/triangulation_test.cc
        )
    add_executable(geometry_test ${GEOMETRY_TEST_FILES})
    target_include_directories(geometry_test PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(geometry_test
        PUBLIC
        geometry
        Eigen3::Eigen
        ${TEST_MAIN})
    add_test(geometry_test geometry_test)
endif()

pybind11_add_module(pygeometry python/pybind.cc)
target_link_libraries(pygeometry
  PRIVATE
    geometry
    foundation
)
set_target_properties(pygeometry PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
