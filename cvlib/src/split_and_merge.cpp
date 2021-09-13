/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>

struct Quadrotree
	{
		std::vector<Quadrotree> childs;
		bool have_childs;
		int i1, i2, j1, j2;

		Quadrotree(int i1 = 0, int i2 = 0, int j1 = 0, int j2 = 0): childs(0), have_childs(false), i1(i1), i2(i2), j1(j1), j2(j2)
		{
		}
		
	};

namespace
{
	
	
void split_image(cv::Mat image, double stddev, Quadrotree *tree)
{
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;
	
	
	if ((height != 1) && (width != 1))
	{
		tree->have_childs = true;
		int dh = (tree->i2 - tree->i1) / 2;
		int dw = (tree->j2 - tree->j1) / 2;
		
		tree->childs.push_back(Quadrotree(tree->i1, tree->i1 + dh, tree->j1, tree->j1 + dw));
		tree->childs.push_back(Quadrotree(tree->i1, tree->i1 + dh, tree->j1 + dw, tree->j2));
		tree->childs.push_back(Quadrotree(tree->i1 + dh, tree->i2, tree->j1, tree->j1 + dw));
		tree->childs.push_back(Quadrotree(tree->i1 + dh, tree->i2, tree->j1 + dw, tree->j2));
		split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev, &(tree->childs[0]));
		split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev, &(tree->childs[1]));
		split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev, &(tree->childs[2]));
		split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev, &(tree->childs[3]));
	}
}

void merge_two(cv::Mat image, double stddev, Quadrotree *tree1, Quadrotree *tree2)
{
	cv::Mat segm1 = image(cv::Range(tree1->i1, tree1->i2), cv::Range(tree1->j1, tree1->j2));
	cv::Mat segm2 = image(cv::Range(tree2->i1, tree2->i2), cv::Range(tree2->j1, tree2->j2));
	
	cv::Mat mean1;
    cv::Mat dev1;
    cv::meanStdDev(segm1, mean1, dev1);
	
	cv::Mat mean2;
    cv::Mat dev2;
    cv::meanStdDev(segm2, mean2, dev2);

    if ((dev1.at<double>(0) <= stddev) && (dev2.at<double>(0) <= stddev))
    {
        segm1.setTo((mean1 + mean2) / 2);
		segm2.setTo((mean1 + mean2) / 2);
        return;
    }
}

void merge_image(cv::Mat image, double stddev, Quadrotree *tree)
{
	while (tree->have_childs)
	{
		bool check_end = !(tree->childs[0].have_childs | tree->childs[1].have_childs | tree->childs[2].have_childs | tree->childs[3].have_childs);
		if (check_end)
		{
			merge_two(image, stddev, &(tree->childs[0]), &(tree->childs[1]));
			merge_two(image, stddev, &(tree->childs[0]), &(tree->childs[2]));
			merge_two(image, stddev, &(tree->childs[3]), &(tree->childs[1]));
			merge_two(image, stddev, &(tree->childs[3]), &(tree->childs[2]));
			tree->have_childs = false;
		}
		else
		{
			merge_image(image, stddev, &(tree->childs[0]));
			merge_image(image, stddev, &(tree->childs[1]));
			merge_image(image, stddev, &(tree->childs[2]));
			merge_image(image, stddev, &(tree->childs[3]));
		}
	}
}



} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    cv::Mat res = image;
	Quadrotree tree(0, image.rows, 0, image.cols);
    split_image(res, stddev, &tree);

    // merge part
    // \todo implement merge algorithm
	merge_image(res, stddev, &tree);
    return res;
}
} // namespace cvlib
