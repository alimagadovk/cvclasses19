/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "utils.hpp"

int demo_corner_detector(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    //auto detector = cv::GFTTDetector::create(); // \todo use cvlib::corner_detector_fast
	auto detector = cvlib::corner_detector_fast::create(); // \todo use cvlib::corner_detector_fast
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
	std::ostringstream convert;

    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);
		
        detector->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);
        // \todo add count of the detected corners at the top left corner of the image. Use green text color.
		convert.str("");
		convert << corners.size() << " corners were found";
		cv::putText(frame, convert.str().c_str(), cv::Point( 10, 25 ), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 150, 0));
        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
