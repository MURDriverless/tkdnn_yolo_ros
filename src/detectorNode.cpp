// include for tkDNN
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "Yolo3Detection.h"

// include for ROS
#include <ros/package.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/String.h"

static const std::string OPENCV_WINDOW = "Image window";
static const int n_classes = 3;
static const int n_batch = 1;

class Detector
{
    public:
        tk::dnn::Yolo3Detection yolo;
        tk::dnn::DetectionNN *detNN;
        std::vector<tk::dnn::box> bbox;

        void imageCallback(
            const sensor_msgs::ImageConstPtr& msg
        );
};

void Detector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        // batch frame will be used for image output
        batch_frame.push_back(cv_ptr->image);

        // dnn input will be resized to network format
        batch_dnn_input.push_back(cv_ptr->image.clone());

        // network inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame);
        
        // show output frame
        cv::imshow("view", batch_frame[0]);
        cv::waitKey(30);

        // update bounding box
        bbox = detNN->batchDetected[0];
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;

    // tkDNN config and initialisation
    // TODO: change to use config file to allow more flexible model switching
    std::string path = ros::package::getPath("tkdnn_yolo_ros");
    std::string net = path + "/src/models/yolo4_cones_int8.rt";

    Detector d;
    d.detNN = &d.yolo;
    d.detNN->init(net, n_classes, n_batch);

    cv::namedWindow("view");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("usb_cam/image_raw", 1,
        &Detector::imageCallback, &d);

    while (ros::ok())
    {
        ros::spinOnce();

        std_msgs::String msg;
        std::stringstream ss;
        ss << "publishing yolo results" << std::endl;
        ss << "num objects detected = " << d.bbox.size() << std::endl;

        for (const auto &b : d.bbox)
        {
            ss << "id:" << b.cl << " prob:" << b.prob << std::endl;
        }

        msg.data = ss.str();
        ROS_INFO("%s", msg.data.c_str());
    }

    cv::destroyWindow("view");

    return 0;
}