#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	Mat image = imread("bookTest.jpg");
	Mat greyImage;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	cvtColor(image, greyImage, COLOR_BGR2GRAY);
	GaussianBlur(greyImage, greyImage, Size(3, 3), 0);
	Canny(greyImage, greyImage, 100, 550);

	findContours(greyImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//this will draw the contours around each item found:
	//drawContours(image, contours, -1, Scalar(0, 255, 0), 2);	

	for (int i = 0; i < contours.size(); ++i)
	{

		double area = contourArea(contours[i]);
		cout << "area of object # " << i << " = " << area << endl;
		
		double length = arcLength(contours[i], true);
		cout << "length of object # " << i << " = " << length << endl;

		Rect rectangle = boundingRect(contours[i]);
		RotatedRect rotateRectangle = minAreaRect(contours[i]);
		
		//storing rectangle vertices. The order is bottomLeft, topLeft, topRight, bottomRight.
		Point2f points[4];
		rotateRectangle.points(points);

		cout << "points of object # " << i << " = ";
		for (int j = 0; j < 4; j++)
		{
			cout << points[j] << ",";
		}
		cout << endl << endl;

		cv::rectangle(image, rectangle, Scalar(0, 0, 255), 2);
	}

	namedWindow("Output", WINDOW_AUTOSIZE);
	imshow("Output", image);

	waitKey(0);
	//imwrite("output.jpg", image);

	return 0;
}