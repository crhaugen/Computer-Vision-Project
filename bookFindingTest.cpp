#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
const int bin = 4;
const int bucketSize = 256 / bin;
// Due to Open CV using BGR values - these are the array index equivalents
const int blue = 0;
const int green = 1;
const int red = 2;


/*
* Create Histogram function:
*	This function creates a color histogram using 3D matrix and looping through an image assigning each of
*	the BGR pixels to each of the histogram buckets.
*		Preconditions: a valid image must be passed in for parsing the pixels for the histogram
*		Postconditions: the histogram is returned after the image has been processed
*/
Mat createHistogram(const Mat& image) {
	// size is a constant - the # of buckets in each dimension
	int dims[3] = { bin, bin, bin };
	// create 3D histogram of integers initialized to zero	
	Mat hist(3, dims, CV_32S, Scalar::all(0));
	// traverse the image and create a histogram of the various colors
	for (int row = 0; row < image.rows - 1; row++) {
		for (int col = 0; col < image.cols - 1; col++) {
			// add the blue pixels to the corresponding histogram bin
			int b = static_cast<int>(image.at<Vec3b>(row, col)[blue] / bucketSize);
			// add the green pixels to the corresponding histogram bin
			int g = static_cast<int>(image.at<Vec3b>(row, col)[green] / bucketSize);
			// add the red pixels to the corresponding histogram bin
			int r = static_cast<int>(image.at<Vec3b>(row, col)[red] / bucketSize);
			// increment the bin by 1
			hist.at<int>(b, g, r)++;
		}
	}
	return hist;
}

/*
* Find Most Common Color Function:
*	This function determines the most common bgr pixel values from the histogram by setting a mostVotes variable
*	and looping through the histogram to find the bin with the most votes and replacing the variable with that
*   value and then determines the most common color by the equation (pixel color) * bucketSize + bucketSize / 2
*		Preconditions: this requires initial BGR values and the histogram to determine the mostVotes in the histogram
*		Postconditions: This function will derive the most common bgr values from the histogram
*/
void findMostCommonColor(int& cBlue, int& cGreen, int& cRed, const Mat& hist) {
	// sets most votes to 0
	int mostVotes = hist.at<int>(0, 0, 0);
	// Loops through each of the histogram bins to determine if that bin has more votes that mostVotes
	for (int i = 0; i < bin; i++) {
		for (int j = 0; j < bin; j++) {
			for (int k = 0; k < bin; k++) {
				// if the bin has the most votes...
				if (hist.at<int>(i, j, k) > mostVotes) {
					// update the most common blue to the value in the first histogram bin
					cBlue = i;
					// update the most common green to the value in the first histogram bin
					cGreen = j;
					// update the most common red to the value in the first histogram bin
					cRed = k;
					// update the mostVotes value and continue looking for the most votes bin
					mostVotes = hist.at<int>(i, j, k);
				}
			}
		}
	}
	// once the looping has completed set the most common red value to r * bucketSize + bucketSize/2;
	cRed = static_cast<int>(cRed * bucketSize + bucketSize / 2);
	// once the looping has completed set the most common green value to g * bucketSize + bucketSize/2;
	cGreen = static_cast<int>(cGreen * bucketSize + bucketSize / 2);
	// once the looping has completed set the most common blue value to b * bucketSize + bucketSize/2;
	cBlue = static_cast<int>(cBlue * bucketSize + bucketSize / 2);
}

/*
* isSolidColor:
*	This function determines if the object is a solid color (rectangle not book) by summing up the total pixels that
*	match the most common color within up to 60% of the pixels and returns true if it's mostly that color.
*		Preconditions: this requires the image for comparison
*		Postconditions: this function produces a boolean for whether or not the object is mostly a solid color.
*/
bool isSolidColor(const Mat& input) {
	// create color histogram
	Mat hist = createHistogram(input);
	// initialize values for most common b,g,r values
	int b = 0;
	int g = 0;
	int r = 0;
	// create an accumulator for the total pixels that match the most common color within a bucketSize
	double total = 0;
	// find the most common color in the histogram of the image
	findMostCommonColor(b, g, r, hist);
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			// store the image color values for comparison and debugging
			int bPrime = static_cast<int>(input.at<Vec3b>(row, col)[blue]);
			int gPrime = static_cast<int>(input.at<Vec3b>(row, col)[green]);
			int rPrime = static_cast<int>(input.at<Vec3b>(row, col)[red]);
			// compare the pixels in the image to the most common color +/- a bucketSize
			if (input.at<Vec3b>(row, col)[blue] - bucketSize <= b && input.at<Vec3b>(row, col)[blue] + bucketSize >= b &&
				input.at<Vec3b>(row, col)[green] - bucketSize <= g && input.at<Vec3b>(row, col)[green] + bucketSize >= g &&
				input.at<Vec3b>(row, col)[red] - bucketSize <= r && input.at<Vec3b>(row, col)[red] + bucketSize >= r) {
					// if so increase the total
					total++;
			}
		}
	}
	// divide the sum by the total number of pixels
	total = total / (input.rows * input.cols);
	if (total > .6)
		return true;
	return false;
}

int main(int argc, char* argv[])
{
	Mat image = imread("bookTest.jpg");
	Mat greyImage;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	cvtColor(image, greyImage, COLOR_BGR2GRAY);
	GaussianBlur(greyImage, greyImage, Size(3, 3), 0);
	Canny(greyImage, greyImage, 100,550);
	
	findContours(greyImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	int bookNumber = 0;

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

		Mat disp(image, rectangle);
		imshow("Display", disp);
		waitKey(0);


		cout << endl << endl;
		// check to see if the detected book is a solid color
		bool sameColor = isSolidColor(disp);
		// if so - don't mark it as a book
		if (sameColor) {
			continue;
		}
		// otherwise
		else {
			// label it as a book and draw the contour
			string tmp = "Book [" + std::to_string(bookNumber) + "]";
			putText(image, tmp, Point(points[1].x + 5, points[1].y - 15), FONT_HERSHEY_COMPLEX_SMALL, .75, Scalar(0, 0, 255), 1.75);
			bookNumber++;
			drawContours(image, contours, bookNumber, Scalar(0, 0, 255), 2);
			//cv::rectangle(image, rectangle, Scalar(0, 0, 255), 2);
		}
	}

	namedWindow("Output", WINDOW_AUTOSIZE);
	imshow("Output", image);
	imwrite("output.jpg", image);
	waitKey(0);

	return 0;
}