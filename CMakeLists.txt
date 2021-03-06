cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cylinder_segmentation)

find_package(PCL 1.7 REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${PNG_INCLUDE_DIR})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (cylinder_seg cylinder_seg.cpp)
target_link_libraries (cylinder_seg ${PCL_LIBRARIES})

add_executable (plane_seg_tester plane_seg_tester.cpp)
target_link_libraries (plane_seg_tester ${PCL_LIBRARIES} ${Boost_LIBRARIES})


add_executable (plane_cylinder_seg plane_cylinder_seg.cpp)
target_link_libraries (plane_cylinder_seg ${PCL_LIBRARIES})

add_executable(seg_tester seg_tester.cpp)
target_link_libraries(seg_tester ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable (online_plane_seg online_plane_seg.cpp)
target_link_libraries (online_plane_seg ${PCL_LIBRARIES} ${PNG_LIBRARY})

add_executable(histogram_match_tester histogram_match_tester.cpp )
target_link_libraries( histogram_match_tester ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(sift sift.cpp )
target_link_libraries(sift ${OpenCV_LIBS}  ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(sift_draw sift_draw.cpp )
target_link_libraries(sift_draw ${OpenCV_LIBS})

add_executable(auto_learn_col auto_learn_col.cpp)
target_link_libraries(auto_learn_col ${PCL_LIBRARIES} ${Boost_LIBRARIES})

find_package(HDF5)
if(HDF5_FOUND)
# find FLANN with pkg-config 
 find_package(PkgConfig REQUIRED) 
 pkg_check_modules(FLANN REQUIRED flann) 
 link_directories(${FLANN_LIBRARY_DIRS}) 
 include_directories(${FLANN_INCLUDE_DIRS}) 
 if(FLANN_CFLAGS) 
   add_definitions(${FLANN_CFLAGS}) 
 endif(FLANN_CFLAGS)
  include_directories(${HDF5_INCLUDE_DIR})

  add_executable(auto_learn_flann auto_learn_flann.cpp)
  target_link_libraries(auto_learn_flann ${PCL_LIBRARIES} ${Boost_LIBRARIES}
                                 ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY})

  add_executable(auto_test_flann auto_test_flann.cpp)
  target_link_libraries(auto_test_flann ${PCL_LIBRARIES} ${Boost_LIBRARIES}
                                 ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY})


endif(HDF5_FOUND)


add_executable(auto_learn_plane auto_learn_plane.cpp)
target_link_libraries(auto_learn_plane ${PCL_LIBRARIES} ${Boost_LIBRARIES})

