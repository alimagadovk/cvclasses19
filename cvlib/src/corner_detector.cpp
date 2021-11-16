/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>
#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

bool check_fragment(cv::Mat &fragment)
{
	int N = 12;
	int threshold = 40;
	unsigned char I1 = std::min((int)fragment.at<unsigned char>(fragment.rows / 2, fragment.cols / 2) + threshold, 255);
	unsigned char I2 = std::max((int)fragment.at<unsigned char>(fragment.rows / 2, fragment.cols / 2) - threshold, 0);
	int i_ind[16] = {0, 3, 6, 3, 0, 0, 1, 2, 4, 5, 6, 6, 5, 4, 2, 1};
	int j_ind[16] = {3, 6, 3, 0, 2, 4, 5, 6, 6, 5, 4, 2, 1, 0, 0, 1};
	int count1 = 0, count2 = 0;
	for (int k = 0; k < 4; k++)
	{
		if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) > I1)
		{
			count1++;
		}
		if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) < I2)
		{
			count2++;
		}
	}
	if ((count1 < 3) && (count2 < 3))
		return false;
	for (int k = 4; k < 16; k++)
	{
		if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) > I1)
		{
			count1++;
		}
		if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) < I2)
		{
			count2++;
		}
	}
	if (((count1 >= N) && (count2 < N)) ^ ((count1 < N) && (count2 >= N)))
		return true;
	else
		return false;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
	keypoints.clear();
	cv::Mat curr_frame;
	image.getMat().copyTo(curr_frame);
	cv::cvtColor(curr_frame, curr_frame, cv::COLOR_BGR2GRAY);
	cv::medianBlur(curr_frame,curr_frame,3);
	
	int border=3;

	cv::copyMakeBorder(curr_frame, curr_frame, border, border, border, border, cv::BORDER_REFLECT_101);
	
	for (int i = border; i < curr_frame.rows - border; i++)
	{
		for (int j = border; j < curr_frame.cols - border; j++)
		{
			cv::Mat &fragment = curr_frame(cv::Range(i - border, i + border + 1), cv::Range(j - border, j + border + 1));
			if (check_fragment(fragment))
			{
				keypoints.push_back(cv::KeyPoint(j, i, 2*border + 1));
			}
		}
	}
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    //std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
