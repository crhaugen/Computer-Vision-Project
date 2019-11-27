#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <algorithm> 


using namespace cv;
using namespace std;
const int bin = 4;
const int bucketSize = 256 / bin;
// Due to Open CV using BGR values - these are the array index equivalents
const int blue = 0;
const int green = 1;
const int red = 2;
const Scalar color = Scalar(0, 0, 255);

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
	if (total > .85)
		return true;
	return false;
}

/*
* findBounds:
*	Function tries to approximative a good threshold to use for finding 
*		edges in any image. Does this based on how light/dark the grayscale image is
*		Preconditions: meanColor contains the mean value for a given image
*		Postconditions: lowerThreshold and upperThreshold contain calculated values.
* 
*/
void findBounds(double meanColor, int &lowerThreshold, int &upperThreshold)
{
	double sigma = 0.33;
	if (meanColor > 195) 
	{
		lowerThreshold = (int)max(double(0), (1 - 2 * sigma) * (255 - meanColor));
		upperThreshold = lowerThreshold * 2;
	}
	else if (meanColor > 130) 
	{
		lowerThreshold = (int)max(double(0), (1 - sigma) * (255 - meanColor));
		upperThreshold = lowerThreshold * 2;
	}
	else if (meanColor < 60) 
	{
		lowerThreshold = (int)max(double(0), (1 - 2 * sigma) * meanColor);
		upperThreshold = lowerThreshold * 2;
	}
	else
	{
		lowerThreshold = (int)max(double(0), (1 - sigma) * meanColor);
		upperThreshold = lowerThreshold * 2;
	}
}

/*
* findMakers:
*	Function takes images that have already been identified as possible books
*		and looks for any makers on the image which could be possible titles or
*		images seen on book covers
*		Postconditions: image is a valid image.
*       Postconditions: true or false of return depending on whether any makers were found.
*/
bool findMarkers(Mat image)
{
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

	morphologyEx(grayImage, grayImage, MORPH_GRADIENT, element);
	threshold(grayImage, grayImage, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

	Mat connected;
	element = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(grayImage, connected, MORPH_CLOSE, element);
	
	Mat mask = Mat::zeros(grayImage.size(), CV_8UC1);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	for (int i = 0; i >= 0; i = hierarchy[i][0])
	{
		//find all the rectangle -ish contours in the image
		Rect rect = boundingRect(contours[i]);
		drawContours(mask, contours, i, Scalar(255, 255, 255), FILLED);
	
		Mat nonZero(mask, rect);
		double ratioOfNonZeroPixel = (double)countNonZero(nonZero) / (double(rect.width) * double(rect.height));

		//try to make it so that it won't be counted as an title/word unless it's at a certain
		//size and has a certain amount of non zero pixels (text) 
		// *These values can be adjusted 
		if (ratioOfNonZeroPixel > .40 && (rect.height > 8 && rect.width > 8))
		{
			rectangle(image, rect, Scalar(0, 255, 0), 2);
			return true;
		}
	}
	return false;
}


int main(int argc, char* argv[])
{
	//Mat image = imread("bookTest.jpg");
	Mat image = imread("bookTest.jpg");
	Mat greyImage;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	
	
	cvtColor(image, greyImage, COLOR_BGR2GRAY);
	GaussianBlur(greyImage, greyImage, Size(3, 3), 0);

	Scalar meanColor = mean(greyImage);
	cout << " " << meanColor[0] << endl;
	
	int lower = 0;
	int upper = 0;
	findBounds(meanColor[0], lower, upper);

	//edged = cv2.Canny(gray_image, lower, upper)
	//cv2.imshow('Edges', edged)

	//threshold(greyImage, greyImage, 0, 255, THRESH_OTSU + THRESH_BINARY);
	Canny(greyImage, greyImage, lower, upper);
	//double threshold = 100;
	//Canny(greyImage, greyImage, threshold, threshold * 2);
	dilate(greyImage, greyImage, Mat(), Point(-1, -1));
	namedWindow("Canny", WINDOW_AUTOSIZE);
	imshow("Canny", greyImage);

	


	imshow("Canny1", greyImage);
	findContours(greyImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	drawContours(image, contours, -1, Scalar(0, 255, 0), 2);
	namedWindow("con", WINDOW_AUTOSIZE);
	imshow("con", image);
	int bookNumber = 0;
	vector<Point> points;

	for (int i = 0; i >= 0; i = hierarchy[i][0])
	{
		
		double area = contourArea(contours[i]);
		cout << "area of object # " << i << " = " << area << endl;

		double epsilon = .02 * arcLength(contours[i], true);

		
		approxPolyDP(contours[i], points, epsilon, true);


		if (points.size() >= 4 && points.size() <= 7)
		{
			cout << "THIS IS A RECTANGLE, SIZE " << points.size() << endl;
		}
		else
		{
			cout << "THIS IS NOT A RECTANGLE, SIZE " << points.size()  << endl;
			continue;
		}


		
		if (area < 2000) {
			continue;
		}

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

		namedWindow("test", WINDOW_AUTOSIZE);
		imshow("test", disp);
		waitKey(0);

		cout << endl << endl;
		
		//after we have found a possible rectangle, do more checks to see
		//if the object has any markers or is just one color.
		if (isSolidColor(disp) || !findMarkers(disp)) {
		    continue;
		}
		// otherwise
		else {
			// label it as a book and draw the contour
			string tmp = "Book [" + std::to_string(bookNumber) + "]";
			putText(image, tmp, Point(points[1].x + 15, points[1].y - 15), FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2.9);
			bookNumber++;
			drawContours(image, contours, bookNumber, color, 2);
			// cv::rectangle(image, rectangle, color, 2);
		}
	}

	namedWindow("Output", WINDOW_AUTOSIZE);
	imshow("Output", image);
	imwrite("output.jpg", image);
	waitKey(0);

	return 0;
}
