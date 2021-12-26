/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
	bool repeat;
	for(int i = 0; i < matches.size(); ++i)
	{
		repeat = true;
		while(repeat)
		{
			repeat = false;
			for(int j = 0; j < (matches[i].size() - 1); ++j)
			{
				for(int l = j + 1; l < matches[i].size(); ++l)
				{
					if ((matches[i][j].distance/matches[i][l].distance) >= ratio_)
					{
						matches[i].erase(matches[i].begin()+l);
						repeat = true;
						break;
					}
				}
			}
		}
	}
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
	if (trainDescCollection.empty())
		return;
	
    auto q_desc = queryDescriptors.getMat();
    auto t_desc = trainDescCollection[0];
    q_desc.convertTo(q_desc, CV_64F);
    t_desc.convertTo(t_desc, CV_64F);
    matches.resize(q_desc.rows);
    for (int i = 0; i < q_desc.rows; ++i)
    {
        for (int j = 0; j < t_desc.rows; ++j)
        {
            double distance = 0;
            for(int ind = 0; ind < t_desc.cols; ++ind)
            {
                distance += (q_desc.at<double>(i,ind) - t_desc.at<double>(j,ind) ) * (q_desc.at<double>(i,ind) - t_desc.at<double>(j,ind));
            }
            if (distance < (maxDistance * maxDistance))
            {
                matches[i].emplace_back(i, j, distance);
            }
        }
		std::sort(matches[i].begin(),matches[i].end(), [](cv::DMatch i,cv::DMatch j) { return (i.distance<j.distance); });
    }
	
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib
