#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <opencv2\imgproc.hpp>
#include <time.h>


using namespace std;
using namespace cv;

//function to rotate Image
Mat rotate(Mat src, double angle)
{
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);

	warpAffine(src, dst, r, Size(src.cols, src.rows), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255, 0.0));
	return dst;
}

int main()
{

	Mat sample;
	sample = imread("C://4.jpg", CV_LOAD_IMAGE_COLOR);
	if (!sample.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	Mat gray;
	// convert RGB image to gray
	cvtColor(sample, gray, CV_BGR2GRAY);

	// morphological gradient
	Mat mg;
	Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(gray, mg, MORPH_GRADIENT, morphKernel);

	// thresholding
	Mat bin;
	threshold(mg, bin, 70, 255.0, THRESH_BINARY);

	// connect regions
	Mat connected;
	morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(bin, connected, MORPH_CLOSE, morphKernel);

	//Hough transform to generate straight lines in image for detecting Rotation
	cv::Size size = connected.size();
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(connected, lines, 1, CV_PI / 180, 50, 50, 10);
	cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));

	double angle = 0.;
	unsigned nb_lines = lines.size();
	for (unsigned i = 0; i < nb_lines; ++i)
	{
		cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
		angle += atan2((double)lines[i][3] - lines[i][1],
			(double)lines[i][2] - lines[i][0]);
	}
	angle /= nb_lines; // mean angle, in radians.

	std::cout << "angle:  " << angle * 180 / CV_PI << std::endl;

	Mat dst;
	dst = rotate(sample, (angle * 180 / CV_PI));
	imshow("rotated Image", dst);// rotated Image

	// convert RGB image to gray
	cvtColor(dst, gray, CV_BGR2GRAY);

	// morphological gradient
    morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	morphologyEx(gray, mg, MORPH_GRADIENT, morphKernel);

	// thresholding
    threshold(mg, bin, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

	// connect horizontally oriented regions
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(bin, connected, MORPH_CLOSE, morphKernel);

	//these two vectors needed for output of findContours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// filter contours
	int i = 0;//pointer for storing cropped out images
	Mat cropped[25];//to store cropped out images
	Mat mask = Mat::zeros(bin.size(), CV_8UC1);
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		Rect rect = boundingRect(contours[idx]);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		// drawing contours
		drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
		// ratio of non-zero pixels in the filled region
		double r = (double)countNonZero(maskROI) / (rect.width*rect.height);

		if (r > .45 &&	(rect.height > 8 && rect.width > 8)) /* constraints on region size */
			
		{
			rectangle(dst, rect, Scalar(0, 255, 0), 2);
			 cropped[i] = dst(rect);
			 i++;
		}
	}

	cout << i << endl;//to check no. of images cropped out
	pyrDown(dst, dst, Size(dst.cols / 2, dst.rows / 2));//to scale down the image

	imshow("Final Result", dst); // final result
	//cropped out images
	imshow("c0", cropped[0]);
	imshow("c1", cropped[1]);
	imshow("c2", cropped[2]);
	imshow("c3", cropped[3]);
	imshow("c4", cropped[4]);
	imshow("c5", cropped[5]);
	imshow("c6", cropped[6]);
	imshow("c7", cropped[7]);
	imshow("c8", cropped[8]);
	imshow("c9", cropped[9]);
	imshow("c10", cropped[10]);
	imshow("c11", cropped[11]);
	imshow("c12", cropped[12]);
	imshow("c13", cropped[13]);
	imshow("c14", cropped[14]);
	imshow("c15", cropped[15]);
	waitKey(0);
	return 0;
}

