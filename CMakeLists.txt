cmake_minimum_required(VERSION 3.0.2)
project(tkdnn_yolo_ros)

# Seems to need to look for OpenCV here, otherwise there is conflict between
# cv_bridge's OpenCV and the Ubuntu OpenCV
# also used catkin config --cmake-args -DOpenCV_DIR=/usr/local/share/OpenCV/
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
  message_generation
)

## Declare ROS messages and services
add_message_files(
  FILES
    BoundingBox.msg
    BoundingBoxes.msg
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
    sensor_msgs
    std_msgs
)

## Declare a catkin package
# catkin_package() 
catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  message_runtime 
  roslib
)

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

# cuda_include_directories(${CUDNN_INCLUDE_DIR})
# cuda_add_executable(${PROJECT_NAME} src/KeypointDetector.cpp src/Utils.cpp src/EntropyCalibrator.cpp)

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


add_library(keypoint src/KeypointDetector.cpp src/Utils.cpp src/EntropyCalibrator.cpp)
target_link_libraries(keypoint ${OPENCV_LIBRARIES} ${catkin_LIBRARIES} nvonnxparser)


#-------------------------------------------------------------------------------
# Executables
#-------------------------------------------------------------------------------
include_directories(include ${catkin_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})

add_executable(detectorNode src/detectorNode.cpp)
target_link_libraries(detectorNode tkDNN keypoint ${OPENCV_LIBRARIES} ${catkin_LIBRARIES})