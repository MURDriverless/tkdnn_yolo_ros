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

Place the TensorRT models (.rt) within `src/models`.
Currently the model used for inferene is defined in `detectorNode.cpp`.