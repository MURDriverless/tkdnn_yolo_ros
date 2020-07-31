# tkDNN YOLO ROS

## Introduction
This is a ROS package/wrapper for using tkDNN and YOLO in a ROS environment.

## Prerequisites

* Nvidia Driver
* CUDA
* cuDNN
* TensorRT

See [tkDNN repo](https://github.com/ceccocats/tkDNN) for detailed information regarding dependencies.

## Setup

Remember to clone in the submodule for `tkDNN`.

```
git clone --recurse-submodules https://github.com/MURDriverless/tkdnn_yolo_rost
```

To get the submodule after `git clone`. ([source](https://stackoverflow.com/questions/16773642/pull-git-submodules-after-cloning-project-from-github))

```
git pull --recurse-submodules
git submodule update --recursive
```

Place the TensorRT models (.rt) within `src/models`.
Currently the model used for inferene is defined in `detectorNode.cpp`.

## Note

The parts for traffic cone keypoints detection were originally forked from https://github.com/cv-core/tensorrt_ros which containts code for inference and format conversion in C++.