/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>
#include <ctime>
#include <random>

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
	
	int border=5;

	cv::copyMakeBorder(curr_frame, curr_frame, border, border, border, border, cv::BORDER_REFLECT_101);
	
	for (int i = border; i < curr_frame.rows - border; i++)
	{
		for (int j = border; j < curr_frame.cols - border; j++)
		{
			cv::Mat &fragment = curr_frame(cv::Range(i - border, i + border + 1), cv::Range(j - border, j + border + 1));
			if (check_fragment(fragment))
			{
				keypoints.push_back(cv::KeyPoint(j - border, i - border, 2*border + 1, 0, 0, 0, 3));
			}
		}
	}
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
}

class Generator
{
	std::default_random_engine generator;
    std::normal_distribution<double> distribution;
    double min;
    double max;
public:
    Generator(double mean, double stddev, double min, double max, int seed = 3):
        distribution(mean, stddev), min(min), max(max)
    {
		generator.seed(seed);
	}

    double operator ()()
	{
        while (true) {
            double number = this->distribution(generator);
            if (number >= this->min && number <= this->max)
                return number;
        }
    }
	
	auto gen_pairs(int pairs_num)
	{
		std::vector<std::pair<int, int>> pairs;
		int x, y;
		for (int i = 0; i < pairs_num; ++i)
		{
			x = (*this)();
			y = (*this)();
			pairs.push_back(std::make_pair(x, y));
		}
		return pairs;
	}
	
};


void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    // \todo implement any binary descriptor
	cv::Mat curr_frame;
	cv::cvtColor(image, curr_frame, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(curr_frame, curr_frame, cv::Size(5, 5), 0, 0);
	
	const int size_ar = keypoints[0].size;
    const int desc_length = 256;
	std::vector<cv::KeyPoint> new_keypoints;
	for (const auto& point : keypoints)
    {
        if ((point.pt.x - size_ar / 2 >= 0) && (point.pt.x + size_ar / 2 < curr_frame.cols) && (point.pt.y - size_ar / 2 >= 0) && (point.pt.y + size_ar / 2 < curr_frame.rows))
        {
            new_keypoints.push_back(point);
        }
    }
	std::cout << "old_keypoints_size: " << keypoints.size()<<"; new_keypoints_size: "<< new_keypoints.size() << std::endl;
	
	Generator gen(0, size_ar/3.0, -size_ar/2, size_ar/2); 
	auto pairs = gen.gen_pairs(desc_length * 2);
	
	descriptors.create(static_cast<int>(new_keypoints.size()), desc_length, CV_32S);
	
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);
	int temp;
	
	for (size_t i = 0; i < desc_mat.rows; i++)
    {
        for (size_t j = 0; j <desc_mat.cols; j++)
        {
			unsigned char x = curr_frame.at<unsigned char>(new_keypoints[i].pt.x + pairs[2 * j].first, new_keypoints[i].pt.y + pairs[2 * j].second);
			unsigned char y = curr_frame.at<unsigned char>(new_keypoints[i].pt.x + pairs[2 * j + 1].first, new_keypoints[i].pt.y + pairs[2 * j + 1].second);
            desc_mat.at<uint>(i, j) = int(x < y);
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& corners, cv::OutputArray descriptors, bool /*= false*/)
{
	this->detect(image,corners);
    this->compute(image, corners,descriptors);
}
} // namespace cvlib
