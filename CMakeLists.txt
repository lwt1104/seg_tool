cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cylinder_segmentation)

find_package(PCL 1.7 REQUIRED)
find_package(PNG REQUIRED)


include_directories(${PNG_INCLUDE_DIR})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (cylinder_seg cylinder_seg.cpp)
target_link_libraries (cylinder_seg ${PCL_LIBRARIES})

add_executable (plane_cylinder_seg plane_cylinder_seg.cpp)
target_link_libraries (plane_cylinder_seg ${PCL_LIBRARIES})

add_executable(seg_tester seg_tester.cpp)
target_link_libraries(seg_tester ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable (online_plane_seg online_plane_seg.cpp)
target_link_libraries (online_plane_seg ${PCL_LIBRARIES} ${PNG_LIBRARY})

