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
*	match the most common color within up to 85% of the pixels and returns true if it's mostly that color.
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
	total = total / ((double)input.rows * (double)input.cols);
	// if the total pixels that match the most common color exceed 85%
	if (total > .85)
		// the object is most likely a solid color (considering lighting, texture, shade, etc.)
		return true;
	// otherwise return false
	return false;
}


/*
* findBooks:
*	Function takes images and tries to identify as many possible books as
*		it can.
*		Preconditions: image is a valid image.
*       Postconditions: vector containing all the possible books found.
*/
vector<Rect> findBooks(Mat& image, vector<vector<Point>>& bookContours)
{
	// create a vector of the bound objects detected as books
	vector<Rect> books;
	Mat grayImage;
	// convert the image to grayscale
	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	// apply the gaussian blur
	GaussianBlur(grayImage, grayImage, Size(3, 3), 2.5, 2.5, 4);
	// retrieve the mean color/intensity from the image
	Scalar meanColor = mean(grayImage);
	// cout << " " << meanColor[0] << endl;

	// loop to set the threshold of an image and detect edges
	for (int i = 55; i < 255; i += 75)
	{
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		// set the lower threshold
		int lowerThreshold = i;
		// set the upper threshold (keep the value within the bounds)
		int upperThreshold = (int)min(lowerThreshold * 3, 255);
		// apply canny filter to the image
		Canny(grayImage, grayImage, lowerThreshold, upperThreshold);
		// apply a gaussian blur to the image
		GaussianBlur(grayImage, grayImage, Size(3, 3), 1.5, 1.5, 2);

		// create a morphology rectangle to be used in dilation of the image
		Mat element = getStructuringElement(MORPH_RECT, Size(4.5, 4.5), Point(1, 1));
		// apply dilation to bolden the edges found
		dilate(grayImage, grayImage, element);

		// find the countours of the image for object and shape recognition
		findContours(grayImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		// create a vector of points for the contours
		vector<Point> points;
		// search through the countours found and stored in the hierarchy array
		for (int i = 0; i >= 0; i = hierarchy[i][0])
		{
			// calculate the area of the given contour
			double area = contourArea(contours[i]);
			// set the epsilon (maximum distance to contour) to 2% 
			double epsilon = .02 * arcLength(contours[i], true);
			// apply contour approximation
			approxPolyDP(contours[i], points, epsilon, true);

			// if it does not have 4 points
			if(points.size() != 4)
			{
				// discard the object
				continue;
			}
			// to reduce false positives from smaller segments of pixels, we set the min. contour area to 1000
			if (area < 1000) {
				continue;
			}
			// We then bind the object detected to a rectangle
			Rect rectangle = boundingRect(contours[i]);
			// create a display image of the object detected
			Mat disp(image, rectangle);
			// add the books to our vector of found books
			books.push_back(rectangle);
			// add the contours to the vector of book contours
			bookContours.push_back(contours[i]);
		}
	}
	// return the vector of books found
	return books;
}


/*
* findMakers:
*	Function takes images that have already been identified as possible books
*		and looks for any makers on the image which could be possible titles or
*		images seen on book covers
*		Preconditions: image is a valid image.
*       Postconditions: true or false of return depending on whether any makers were found.
*/
bool findMarkers(Mat image)
{
	// create a grayscale image
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	// create an elipse element
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	// create morphology object to detect structures within the image
	morphologyEx(grayImage, grayImage, MORPH_GRADIENT, element);
	// apply a threshold to the colors/intensity of pixels within the image
	threshold(grayImage, grayImage, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

	// create an image for detecting the objects that are rectangular within the image
	Mat connected;
	// create a morphology element to detect rectangles within the photo
	element = getStructuringElement(MORPH_RECT, Size(9, 1));
	// find the rectangular (closed) objects within the image
	morphologyEx(grayImage, connected, MORPH_CLOSE, element);
	// create a new  matrix for the mask to 
	Mat mask = Mat::zeros(grayImage.size(), CV_8UC1);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	// find the contours within the image
	findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int i = 0; i >= 0; i = hierarchy[i][0])
	{
		//find all the rectangle -ish contours in the image
		Rect rect = boundingRect(contours[i]);
		// draw the contours for the image
		drawContours(mask, contours, i, Scalar(255, 255, 255), FILLED);

		// create a non-zero mask for detecting markers within the image
		Mat nonZero(mask, rect);
		// calculate the ratio of non-zero pixels within the entire image
		double ratioOfNonZeroPixel = (double)countNonZero(nonZero) / (double(rect.width) * double(rect.height));

		//try to make it so that it won't be counted as an title/word unless it's at a certain
		//size and has a certain amount of non zero pixels (text) 
		if (ratioOfNonZeroPixel > .40 && (rect.height > 8 && rect.width > 8))
		{
			// create a rectangle around the markers found within the image
			rectangle(image, rect, Scalar(0, 255, 0), 2);
			// return true for the marker being found
			return true;
		}
	}
	// otherwise the markers were not found - return false
	return false;
}

/*
* bookAlreadyFound:
*	Function takes a vector that has the location of books and check it with
*		new points to see if we already got this book
*		images seen on book covers
*		Preconditions: booksFound contains points of books, x and y are valid points
*       Postconditions: true or false of return depending on whether the book has already been found.
*/
bool bookAlreadyFound(vector<Point> booksFound, int x, int y)
{
	// for each of the books found in the image
	for (int i = 0; i < booksFound.size(); i++)
	{
		// check to see if the points have alread been found within the image
		if ((booksFound[i].x - 10) < x && x < (booksFound[i].x + 10))
		{
			if ((booksFound[i].y - 10) < y && y < (booksFound[i].y + 10))
			{
				// return true if the book exists within our book vector
				return true;
			}
		}
	}
	// otherwise it is a new book
	return false;
}

/*
* Main:
*	The primary function will search the current program directory to find all .jpg images and search for books in 
*		each image. We decided to add in the directory search functionality to reduce naming errors and demonstration
*		purposes. This applies the algorithms mentioned above to produce output images of each of the test photos
*		Preconditions: This can be run either directly or with use of the bash script
*       Postconditions: Output images with books detected are produced with the images found in the directory
*/
int main(int argc, char* argv[])
{
	// create a vector of directory images;
	vector<String> directoryImages;

	// set the directory location to the current folder, find photos and add to directoryImages
	glob("./*.jpg", directoryImages, false);
	// if directory is empty, notify user and close program
	if (directoryImages.size() == 0) {
		cout << "Error - no images were detected. Make sure the file extensions are '.jpg' format "<< 
					"and try again." << endl;
		return 0;
	}
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
		// create a number for the book
		int bookNumber = 1;
		// read in the image from the directory
		Mat image = imread(directoryImages[i]);
		// create a contour vector for the points
		vector<vector<Point>> contours;
		// create the initial image for each file it scans (image1.jpg, image2.jpg ... etc.)
		string initialImage = "image" + std::to_string(i + 1);
		// create a window to display the image
		namedWindow(initialImage, WINDOW_AUTOSIZE);
		// display the image
		imshow(initialImage, image);
		// display the image until user closes the window
		waitKey(0);
		// find the books within the image
		vector<Rect> books = findBooks(image, contours);
		// create a vector for the coordinates where the books were found
		vector<Point> foundBooks;
		// for each of the objects identified as a book
		for (int i = 0; i < books.size(); i++)
		{
			// create an image of the current book being examined
			Mat disp(image, books[i]);
			// check to see if it is a solid color or doesn't have markers
			if (isSolidColor(disp) || !findMarkers(disp))
			{
				// if it lacks the descriptors, it is a false positive, continue
				continue;
			}
			// otherwise check if we have already found the book
			else
			{
				// if the object detected at that location has not already been defined as a found book
				if (!bookAlreadyFound(foundBooks, books[i].x, books[i].y))
				{
					// draw the contour around the book
					drawContours(image, contours, i, Scalar(0, 255, 0), 2);
					// add the book to our collection of found books
					foundBooks.push_back(Point(books[i].x, books[i].y));
					// create a string of text to output the book number
					string bookNum = "Book [" + std::to_string(bookNumber) + "]";
					// output the book number above the 
					putText(image, bookNum, Point(books[i].x + 1, books[i].y + 20), FONT_HERSHEY_COMPLEX_SMALL, .9, color, 2.9);
					// increment the bookNumber
					bookNumber++;
				}
			}
		}
		// create the output image for each file it scans (output1.jpg, output2.jpg ... etc.)
		string name = "output" + std::to_string(i + 1);
		// create a window to display the image
		namedWindow(name, WINDOW_AUTOSIZE);
		// display the image
		imshow(name, image);
		// write the output image to the file directory
		imwrite(name + ".jpg", image);
		// display the image until user closes the window
		waitKey(0);
	}
	// clear the directory of images to free the memory
	directoryImages.clear();
	// return successful
	return 0;
}