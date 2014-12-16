/*
Description	:	Multiple shapes detection, with image.
Source			:	http://opencv-srf.blogspot.com/2011/09/object-detection-tracking-using-contours.html
*/

#include"Utility.h"

/*
- greyscale only
*/

#ifdef SAMPLE_MULTIPLESHAPES
int main()
#else
int sample_multiple_shapes_main()
#endif
{

	IplImage* img = cvLoadImage("Res\\image\\FindingContours_Multi.png");

	//show the original image
	cvNamedWindow("Raw");
	cvShowImage("Raw", img);

	//converting the original image into grayscale
	IplImage* imgGrayScale = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, imgGrayScale, CV_BGR2GRAY);

	//thresholding the grayscale image to get better results
	cvThreshold(imgGrayScale, imgGrayScale, 128, 255, CV_THRESH_BINARY);

	CvSeq* contours;  //hold the pointer to a contour in the memory block
	CvSeq* result;   //hold sequence of points of a contour
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours

	//finding all contours in the image
	cvFindContours(imgGrayScale, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//iterating through each contour
	while (contours)
	{
		//obtain a sequence of points of contour, pointed by the variable 'contour'
		result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

		//if there are 3  vertices  in the contour(It should be a triangle)
		if (result->total == 3)
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

		//if there are 4 vertices in the contour(It should be a quadrilateral)
		else if (result->total == 4)
		{
			//iterating through each point
			CvPoint *pt[4];
			for (int i = 0; i<4; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the quadrilateral
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 255, 0), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 255, 0), 4);
			cvLine(img, *pt[2], *pt[3], cvScalar(0, 255, 0), 4);
			cvLine(img, *pt[3], *pt[0], cvScalar(0, 255, 0), 4);
		}

		//if there are 7  vertices  in the contour(It should be a heptagon)
		else if (result->total == 7)
		{
			//iterating through each point
			CvPoint *pt[7];
			for (int i = 0; i<7; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the heptagon
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[2], *pt[3], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[3], *pt[4], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[4], *pt[5], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[5], *pt[6], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[6], *pt[0], cvScalar(0, 0, 255), 4);
		}

		//obtain the next contour
		contours = contours->h_next;
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