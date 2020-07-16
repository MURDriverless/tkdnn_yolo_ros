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

static const std::string OPENCV_WINDOW = "Image window";
static const int n_classes = 3;
static const int n_batch = 1;

void imageCallback(const sensor_msgs::ImageConstPtr& msg, tk::dnn::DetectionNN *net)
{
    cv_bridge::CvImagePtr cv_ptr;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        batch_frame.push_back(cv_ptr->image);

        // dnn input will be resized to network format
        batch_dnn_input.push_back(cv_ptr->image.clone());

        // network inference
        net->update(batch_dnn_input, n_batch);
        net->draw(batch_frame);
        
        cv::imshow("view", batch_frame[0]);
        cv::waitKey(30);
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

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN *detNN;
    detNN = &yolo;
    detNN->init(net, n_classes, n_batch);

    cv::namedWindow("view");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("usb_cam/image_raw", 1, 
        boost::bind(imageCallback, _1, detNN));
    ros::spin();
    cv::destroyWindow("view");

    return 0;
}


// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "detectorNode");
//     ros::NodeHandle n;
//     ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
//     ros::Rate loop_rate(10);

//     // construct path to the configs

//     // grab image from an image stream
//     image_transport::ImageTransport it(n);
//     image_transport::Subscriber sub = it.subscribe("usb_cam/image_raw", 1, imageCallback);
    
//     // image_transport::Subscriber sub = it.subscribe("jetbot_camera/raw", 1,
//     //     [&detector, &res](const sensor_msgs::ImageConstPtr &msg) -> void { imageCallback(msg, detector, res); });

//     // cv::Mat mat_image = cv::imread(img_path, cv::IMREAD_UNCHANGED);

//     // auto t0 = std::chrono::high_resolution_clock::now();
//     // detector.detect(mat_image, res);
//     // auto t1 = std::chrono::high_resolution_clock::now();

//     // std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);

//     /**
// 	 * A count of how many messages we have sent. This is used to create
// 	 * a unique string for each message.
// 	 */
//     int count = 0;
//     while (ros::ok())
//     {
// 	/**
//      * This is a message object. You stuff it with data, and then publish it.
//      */
//         std_msgs::String msg;

//         std::stringstream ss;
//         ss << "publishing yolo results " << count << std::endl;
//         // ss << "inference duration = " << time_span.count() << " seconds." << std::endl;

//         // DEBUG
//         // ss << cfg_path << std::endl;
//         // ss << weights_path << std::endl;
//         // ss << img_path << std::endl;

//         // publish detection results
//         for (const auto &r : res)
//         {
//             ss << "id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
//         }

//         msg.data = ss.str();

//         ROS_INFO("%s", msg.data.c_str());

//         chatter_pub.publish(msg);

//         ros::spinOnce();

//         loop_rate.sleep();
//         ++count;
//     }

//     return 0;
// }