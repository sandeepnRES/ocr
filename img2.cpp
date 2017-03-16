#pragma package <opencv>
#include <iostream>
#include <unistd.h>
#include <time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>
#include<vector>
#include"segmentation.cpp"
using namespace cv;
using namespace std;


struct hsvRange	{
	int LowH;	int HighH;

	int LowS; 	int HighS;

	int LowV;	int HighV;
};
//colours

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;
struct hsvRange blue={0,100,0,100,0,100};

Mat thresH(char *file,char *mode)
{
    int start, end,diff,l=0;
const int MIN_CONTOUR_AREA = 5;



    Mat imgOriginal=imread( file, CV_LOAD_IMAGE_GRAYSCALE  ),imgOrig,matBlurred, imgThresholded,tmp,t2;

    Mat imgHSV;
    namedWindow("imgOriginal", CV_WINDOW_AUTOSIZE);
    namedWindow("imgThresh", CV_WINDOW_AUTOSIZE);
    imshow("imgOriginal",imgOriginal);

        cv::GaussianBlur(imgOriginal,			// input image
		    matBlurred,							// output image
		    cv::Size(1,1),						// smoothing window width and height in pixels
		    0);					// sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
imwrite("GausianBlur.png",matBlurred);
imshow("BlurredImg",matBlurred);
    waitKey(0);
    threshold (matBlurred,imgThresholded,0,255,CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
    imwrite("thresh.png",imgThresholded);
    imshow("imgThresh",imgThresholded);
    waitKey(0);
    	
    segment(imgThresholded,imgOriginal,mode);
    
    return imgThresholded;
}
int main( int argc, char** argv )
{
	
	Point p[20];int k=0;
	Mat detect;
	detect=thresH(argv[1],argv[2]);
}
