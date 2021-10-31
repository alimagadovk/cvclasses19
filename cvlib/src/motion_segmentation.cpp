/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <deque>
#include <vector>
#include <iostream>

namespace cvlib
{
void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double)
{
    // \todo implement your own algorithm:
    //       * MinMax
    //       * Mean
    //       * 1G
    //       * GMM
	static int count = 0;
	static bool show_flag = false;
	static std::vector<cv::Mat> frames;
	const int N = 5;

	cv::Mat curr_frame = _image.getMat();
	
	cv::medianBlur(curr_frame,curr_frame,3);
	
	cv::Mat D(curr_frame.size(), curr_frame.type());
	cv::Mat M(curr_frame.size(), curr_frame.type());
	cv::Mat m(curr_frame.size(), curr_frame.type());
	
	
	static cv::Mat last_frame = curr_frame * 0;
	
	
	if (count < N)
	{
		frames.push_back(curr_frame);
		count++;
		return;
	}
	else
	{
		frames.erase(frames.begin());
		frames.push_back(curr_frame);
	}
	
	
	D.setTo(0);
	frames.back().copyTo(M);
	frames.back().copyTo(m);
	for (int k = 0; k < N - 1; k++)
	{
		cv::absdiff(frames[k], frames[k + 1], D);
		for (int i = 0; i < curr_frame.size().height; i++)
		{
			for (int j = 0; j < curr_frame.size().width; j++)
			{
				M.at<unsigned char>(i, j) = (frames[k].at<unsigned char>(i, j) > M.at<unsigned char>(i, j)) ?  frames[k].at<unsigned char>(i, j) : M.at<unsigned char>(i, j);
				m.at<unsigned char>(i, j) = (frames[k].at<unsigned char>(i, j) < m.at<unsigned char>(i, j)) ?  frames[k].at<unsigned char>(i, j) : m.at<unsigned char>(i, j);
			}
		}
	}
	
	cv::Mat F(curr_frame.size(), curr_frame.type());
	
	auto temp_D = D.reshape(0,1); // spread Input Mat to single row
	std::vector<unsigned char> vecFromMat;
	temp_D.copyTo(vecFromMat);
    std::nth_element(vecFromMat.begin(), (vecFromMat.begin() + vecFromMat.size()/2), vecFromMat.end());
    unsigned char median = vecFromMat[vecFromMat.size()/2];
	
	
	cv::Mat M_diff, m_diff, M_res, m_res, res;
	
	cv::absdiff(curr_frame, M, M_diff);
	cv::absdiff(curr_frame, m, m_diff);
	
	cv::compare(M_diff, cv::Scalar(median * threshold), M_res, cv::CMP_GT);
	cv::compare(m_diff, cv::Scalar(median * threshold), m_res, cv::CMP_GT);
	
	
	res = M_res | m_res;
	res = res * 255;
	
	res.copyTo(_fgmask);
	
	last_frame = curr_frame;
			
    // \todo implement bg model updates
}

void motion_segmentation::setVarThreshold(double varThreshold)
{
    threshold = varThreshold;
}

} // namespace cvlib
