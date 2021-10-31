/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
	
	std::vector<cv::Mat> kernels; 
    std::vector<cv::Mat> feature_maps;
	
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }
	
	double norm_l2() const
	{
		double res(0.0);
		for (auto v : *this)
		{
			res += v * v;
		}
		return std::sqrt(res);
	}
};

void calculateDescriptor(const cv::Mat& image, descriptor& descr)
{
    descr.clear();
    cv::Mat response;
    cv::Mat mean;
    cv::Mat dev;

    for (const auto& kernel : kernels)
    {
        cv::filter2D(image, response, CV_32F, kernel);
        cv::meanStdDev(response, mean, dev);
        descr.emplace_back(mean.at<double>(0));
        descr.emplace_back(dev.at<double>(0));
    }
}

void calculateDescriptor(const cv::Rect& roi, descriptor& descr)
{   
    descr.clear();
    cv::Mat mean;
    cv::Mat dev;
    for(const auto& map : feature_maps)
    {
        cv::meanStdDev(map(roi), mean, dev);
        descr.emplace_back(mean.at<double>(0));
        descr.emplace_back(dev.at<double>(0));
    }
}

void calculateFilters(int kernel_size)
{
    kernels.clear();
    const double th = CV_PI / 4;
    const double lm = 10.0;
    const double gm = 0.75;
	const double sg = 2;

    // \todo implement complete texture segmentation based on Gabor filters
    // (find good combinations for all Gabor's parameters)
	for (double g = 0; g <= gm; g += (gm / 3))
	{
		for (double l = 1; l <= lm; l += (lm / 2))
		{
			for (double t = 0; t <= th; t += th)
			{
				for (double s = 1; s <= sg; s++)
				{
					kernels.push_back(cv::getGaborKernel(cv::Size(kernel_size, kernel_size), s, t, l, g));
				}
			}
		}
	}
}

void calculateMaps(const cv::Mat& image)
{
    if(kernels.empty()) return;

    feature_maps.clear();
    for (const auto& kernel : kernels)
    {
        feature_maps.emplace_back();
        cv::filter2D(image, feature_maps.back(), CV_32F, kernel);
    }
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
	cv::medianBlur(image,image,3); // pre-filtration for getting better segmentation result
    cv::Mat imROI = image(roi);

	
    //const int kernel_size = ((std::min(roi.height, roi.width) / 2) % 2 == 0) ? (std::min(roi.height, roi.width) - 1) : std::min(roi.height, roi.width); // \todo round to nearest odd
	const int kernel_size = 12;
	static int last_kernel_size;
	
	if (last_kernel_size != kernel_size) 
    {
        calculateFilters(kernel_size);
        last_kernel_size = kernel_size;
    }
	calculateMaps(image);
    descriptor reference;
    calculateDescriptor(image(roi), reference);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();
	
	int dy = kernel_size;
	int dx = kernel_size;

    // \todo move ROI smoothly pixel-by-pixel
    for (int i = 0; i < image.size().width - roi.width; i += dy)
    {
        for (int j = 0; j < image.size().height - roi.height; j += dx)
        {
            auto curROI = baseROI + cv::Point(i,j);
            calculateDescriptor(curROI, test);
            // \todo implement and use norm L2
            res(curROI) = 255 * ((test - reference).norm_l2() <= eps);
        }
    }

    return res;
}
} // namespace cvlib
