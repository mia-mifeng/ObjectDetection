/*
Description	:	Triangle detection, with image.
Source			:	http://opencv-srf.blogspot.com/2011/09/object-detection-tracking-using-contours.html
*/

#include"Utility.h"

#ifdef SAMPLE_TRIANGLE
int main()
#else
int sample_triangle_main()
#endif
{

	IplImage* img = cvLoadImage("Res\\image\\DetectingContours.jpg");

	//show the original image
	cvNamedWindow("Original");
	cvShowImage("Original", img);

	//smooth the original image using Gaussian kernel to remove noise
	cvSmooth(img, img, CV_GAUSSIAN, 3, 3);

	//converting the original image into grayscale
	IplImage* imgGrayScale = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, imgGrayScale, CV_BGR2GRAY);

	cvNamedWindow("GrayScale Image");
	cvShowImage("GrayScale Image", imgGrayScale);

	//thresholding the grayscale image to get better results
	cvThreshold(imgGrayScale, imgGrayScale, 100, 255, CV_THRESH_BINARY_INV);

	cvNamedWindow("Thresholded Image");
	cvShowImage("Thresholded Image", imgGrayScale);

	CvSeq* contour;  //hold the pointer to a contour
	CvSeq* result;   //hold sequence of points of a contour
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours

	//finding all contours in the image
	cvFindContours(imgGrayScale, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//iterating through each contour
	while (contour)
	{
		//obtain a sequence of points of the countour, pointed by the variable 'countour'
		result = cvApproxPoly(contour, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contour)*0.02, 0);

		//if there are 3 vertices  in the contour and the area of the triangle is more than 100 pixels
		if (result->total == 3 && fabs(cvContourArea(result, CV_WHOLE_SEQ))>100)
		{
			//iterating through each point
			CvPoint *pt[3];
			for (int i = 0; i<3; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the triangle
			cvLine(img, *pt[0], *pt[1], cvScalar(255, 0, 0), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(255, 0, 0), 4);
			cvLine(img, *pt[2], *pt[0], cvScalar(255, 0, 0), 4);

		}

		//obtain the next contour
		contour = contour->h_next;
	}

	//show the image in which identified shapes are marked   
	cvNamedWindow("Tracked");
	cvShowImage("Tracked", img);

	cvWaitKey(0); //wait for a key press

	//cleaning up
	cvDestroyAllWindows();
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
	cvReleaseImage(&imgGrayScale);

	return 0;
}