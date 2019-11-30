#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
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
	if (total > .95)
		return true;
	return false;
}


/*
* findBooks:
*	Function takes images and tries to identify as many possible books as
*		it can.
*		Postconditions: image is a valid image.
*       Postconditions: vector containing all the possible books found.
*/
vector<Rect> findBooks(Mat& image, vector<vector<Point>>& bookContours)
{

	vector<Rect> books;
	Mat grayImage;

	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	GaussianBlur(grayImage, grayImage, Size(3, 3), 2.5, 2.5, 4);

	Scalar meanColor = mean(grayImage);
	cout << " " << meanColor[0] << endl;


	for (int i = 55; i < 255; i += 75)
	{
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		int lowerThreshold = i;
		int upperThreshold = (int)min(lowerThreshold * 3, 255);
		Canny(grayImage, grayImage, lowerThreshold, upperThreshold);
		GaussianBlur(grayImage, grayImage, Size(3, 3), 1.5, 1.5, 2);

		Mat element = getStructuringElement(MORPH_RECT, Size(4.5, 4.5), Point(1, 1));
		dilate(grayImage, grayImage, element);
		// namedWindow("Canny", WINDOW_AUTOSIZE);
		// imshow("Canny", grayImage);


		findContours(grayImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		// namedWindow("con", WINDOW_AUTOSIZE);
		// imshow("con", image);


		vector<Point> points;

		for (int i = 0; i >= 0; i = hierarchy[i][0])
		{

			double area = contourArea(contours[i]);
			cout << "area of object # " << i << " = " << area << endl;

			double epsilon = .02 * arcLength(contours[i], true);

			approxPolyDP(contours[i], points, epsilon, true);


			if (points.size() == 4)
			{
				cout << "THIS IS A RECTANGLE, SIZE " << points.size() << endl;
			}
			else
			{
				cout << "THIS IS NOT A RECTANGLE, SIZE " << points.size() << endl;
				continue;
			}

			if (area < 2000) {
				continue;
			}

			double length = arcLength(contours[i], true);
			cout << "length of object # " << i << " = " << length << endl;


			Rect rectangle = boundingRect(contours[i]);
			Mat disp(image, rectangle);
			books.push_back(rectangle);
			bookContours.push_back(contours[i]);
		}
	}

	return books;
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

/*
* bookAlreadyFound:
*	Function takes a vector that has the location of books and check it with
*		new points to see if we already got this book
*		images seen on book covers
*		Postconditions: booksFound contains points of books, x and y are valid points
*       Postconditions: true or false of return depending on whether the book has already been found.
*/
bool bookAlreadyFound(vector<Point> booksFound, int x, int y)
{
	for (int i = 0; i < booksFound.size(); i++)
	{
		if ((booksFound[i].x - 10) < x && x < (booksFound[i].x + 10))
		{
			if ((booksFound[i].y - 10) < y && y < (booksFound[i].y + 10))
			{
				return true;
			}
		}
	}

	return false;
}


int main(int argc, char* argv[])
{
	// create a vector of directory images;
	vector<String> directoryImages;
	string directoryLocation;

	// set the directory location to the current folder
	glob("./*.jpg", directoryImages, false);
	// lambda to remove the output images so it doesn't infinitely create images
	directoryImages.erase(
		// remove if the directory name begins with .\output
		remove_if(directoryImages.begin(), directoryImages.end(),
			[](const std::string& s) {return s.find("output") != string::npos; }
		),
		directoryImages.end()
	);
	// vector to hold the images found in current directory
	vector<Mat> images;
	//number of jpg files in images folder
	int count = directoryImages.size();

	// for each image in the directory look for books
	for (int i = 0; i < count; i++) {

		Mat image = imread(directoryImages[i]);
		// Mat image = imread("bookTest#5.jpg");
		vector<vector<Point>> contours;
		vector<Rect> books = findBooks(image, contours);

		int bookNumber = 1;
		vector<Point> foundBooks;

		for (int i = 0; i < books.size(); i++)
		{
			Mat disp(image, books[i]);

			if (isSolidColor(disp) || !findMarkers(disp))
			{
				continue;
			}
			else
			{
				if (!bookAlreadyFound(foundBooks, books[i].x, books[i].y))
				{
					drawContours(image, contours, i, Scalar(0, 255, 0), 2);
					foundBooks.push_back(Point(books[i].x, books[i].y));
					cout << "book x " << books[i].x;
					cout << "book y " << books[i].y;

					string bookNum = "Book [" + std::to_string(bookNumber) + "]";
					putText(image, bookNum, Point(books[i].x + 1, books[i].y + 20), FONT_HERSHEY_COMPLEX_SMALL, .9, color, 2.9);
					bookNumber++;
				}
			}
		}
		string name = "output" + std::to_string(i + 1);
		namedWindow(name, WINDOW_AUTOSIZE);
		imshow(name, image);
		imwrite(name + ".jpg", image);
		waitKey(0);
	}
	directoryImages.clear();
	return 0;
}
