cmake_minimum_required(VERSION 3.0.2)
project(tkdnn_yolo_ros)

# # http://sirien.metz.supelec.fr/depot/SIR/TutoROS/Codes/CMakeLists.txt.html
# ## Find pkg_config
# find_package(PkgConfig REQUIRED)

# ## Use pkg_config to recover some dependencies
# # The following defines the variables
# # OPENCV_CFLAGS_OTHER, OPENCV_INCLUDE_DIRS, OPENCV_LIBRARIES, ...,
# pkg_check_modules(OPENCV REQUIRED opencv)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  roslib
  sensor_msgs
  cv_bridge
  image_transport
)

## Declare ROS messages and services
# add_message_files(FILES Num.msg)
# add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
# generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-unused-variable")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/include/tkDNN)

add_definitions(-DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN")

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)

find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

# compile
file(GLOB tkdnn_CUSRC "src/tkDNN/src/kernels/*.cu" "src/tkDNN/src/sorting.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

# gives problems in cross-compiling, probably malformed cmake config
#find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/tkDNN/src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})

#-------------------------------------------------------------------------------
# Executables
#-------------------------------------------------------------------------------
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})

add_executable(detectorNode src/detectorNode.cpp)
target_link_libraries(detectorNode tkDNN ${OPENCV_LIBRARIES})