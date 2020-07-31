// include for tkDNN
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "Yolo3Detection.h"

// include for keypoint detector
#include "KeypointDetector.h"
#include <algorithm>

#include <tkdnn_yolo_ros/BoundingBoxes.h>
#include <tkdnn_yolo_ros/BoundingBox.h>

#include <geometry_msgs/Point.h>

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
        // TODO: refactor class
        tk::dnn::Yolo3Detection yolo;
        tk::dnn::DetectionNN *detNN;
        std::vector<tk::dnn::box> bboxs;


        Detector();
        void drawKeypoints(cv::Mat &img, tkdnn_yolo_ros::BoundingBoxes boxes);
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    private:
        std::unique_ptr<KeypointDetector> keypointDetector_;
        int keypointsW;
        int keypointsH;
        int maxBatch;
};

Detector::Detector()
{
    keypointsW = 80;
    keypointsH = 80;
    maxBatch = 100;
    
    keypointDetector_.reset(
        new KeypointDetector(
            ros::package::getPath("tkdnn_yolo_ros") + "/src/models/best_keypoints.onnx", 
            ros::package::getPath("tkdnn_yolo_ros") + "/src/models/best_keypoints.trt", 
            keypointsW, 
            keypointsH, 
            maxBatch)
    );
}

void Detector::drawKeypoints(cv::Mat &img, tkdnn_yolo_ros::BoundingBoxes boxes)
{
    for (auto &b : boxes.bounding_boxes)
    {
        for (auto &pt : b.keypoints)
        {
            cv::circle(img, cv::Point2f(pt.x, pt.y), 3, cv::Scalar(0, 255, 0), -1, 8);
        }
    }
}

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

        // update bounding box
        bboxs = detNN->batchDetected[0];

        tkdnn_yolo_ros::BoundingBoxes boxes;

        // generate a vector of image crops for keypoint detector
        std::vector<cv::Mat> rois;

        for (const auto &bbox: bboxs)
        {
            int left    = std::max(double(bbox.x), 0.0);
            int right   = std::min(double(bbox.x + bbox.w), double(cv_ptr->image.cols));
            int top     = std::max(double(bbox.y), 0.0);
            int bot     = std::min(double(bbox.y + bbox.h), double(cv_ptr->image.rows));

            cv::Rect box(cv::Point(left, top), cv::Point(right, bot));
            cv::Mat roi = cv_ptr->image(box);
            rois.push_back(roi);

            tkdnn_yolo_ros::BoundingBox boundingBox;
            boundingBox.probability = bbox.prob;
            boundingBox.xmin = left;
            boundingBox.ymin = top;
            boundingBox.xmax = right;
            boundingBox.ymax = bot;
            boundingBox.Class = "cone"; // TODO: assign proper colour

            boxes.bounding_boxes.push_back(boundingBox);
        }

        // keypoint network inference
        std::vector<std::vector<cv::Point2f>> keypoints = keypointDetector_->doInference(rois);

        std::cout << "num keypoint sets = " << keypoints.size() << std::endl;
        std::cout << "num bounding boxes = " << boxes.bounding_boxes.size() << std::endl;
        // std::cout << "num keypoint sets = " << keypoints.size() << std::endl;

        // post process keypoints results
        for (unsigned int i = 0; i < boxes.bounding_boxes.size(); ++i)
        {
            std::cout << "i = " << i << ", keypoints = " << keypoints[i].size() << std::endl;
            
            for (auto pt : keypoints[i])
            {
                geometry_msgs::Point point;
                point.x = boxes.bounding_boxes[i].xmin + pt.x;
                point.y = boxes.bounding_boxes[i].ymin + pt.y;
                point.z = 0;

                std::cout << pt.x << "   " << pt.y << std::endl;

                boxes.bounding_boxes[i].keypoints.push_back(point);
            }
        }
        
        detNN->draw(batch_frame);
        drawKeypoints(batch_frame[0], boxes);
        
        // show output frame
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

    // parse parameters


    // tkDNN config and initialisation
    // TODO: change to use config file to allow more flexible model switching
    std::string path = ros::package::getPath("tkdnn_yolo_ros");
    std::string net = path + "/src/models/yolo4_cones_int8.rt";

    Detector d;
    d.detNN = &d.yolo;
    d.detNN->init(net, n_classes, n_batch);

    // keypoint detector config and initialisation
    

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
        ss << "num objects detected = " << d.bboxs.size() << std::endl;

        for (const auto &b : d.bboxs)
        {
            ss << "id:" << b.cl << " prob:" << b.prob << std::endl;
        }

        msg.data = ss.str();
        // ROS_INFO("%s", msg.data.c_str());
    }

    cv::destroyWindow("view");

    return 0;
}