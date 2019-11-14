#include <opencv2/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// Purpose: Compares a template image against a search image.
// Preconditions: search and template images must be grayscale, and match_method must be valid
// Postconditions: search image with rectangle drawn around the found template is returned
// ***Testing was done using search.jpg/template.jpg (find man's face) and search1.jpg/template1.jpg (find waldo)
// ***Future extensions include adding a trackbar with different match methods, so that the result of
// each match method can be compared to find which match method is the best
Mat matchTemplate(const Mat &search, const Mat &templ, int match_method) {

	// map of comparison results--single channel, 32 bit floating point
	int result_cols = search.cols - templ.cols + 1;
	int result_rows = search.rows - templ.rows + 1;
	Mat result(result_rows, result_cols, CV_32FC1);

	matchTemplate(search, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double min_val;
	double max_val;
	Point min_Idx;
	Point max_Idx;
	minMaxLoc(result, &min_val, &max_val, &min_Idx, &max_Idx, Mat());

	Point matchIdx;
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) {
		matchIdx = min_Idx;
	}
	else {
		matchIdx = max_Idx;
	}

	Mat search_copy(search);
	rectangle(search_copy, matchIdx, Point(matchIdx.x + templ.cols, matchIdx.y + templ.rows), Scalar::all(0), 2, 8, 0);

	return search_copy;
}

int main(int argc, char* argv[]) {
	Mat search = imread("search1.jpg", IMREAD_GRAYSCALE);
	Mat templ = imread("template1.jpg", IMREAD_GRAYSCALE);

	int match_method = TM_CCOEFF_NORMED;
	Mat template_match = matchTemplate(search, templ, match_method);

	imshow("template_match", template_match);
	waitKey(0);
}