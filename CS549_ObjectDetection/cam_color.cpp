#include"Utility.h"
#include <iostream>

using namespace cv;
using namespace std;

#ifdef COLOR_PRE
int main(int argc, char** argv)
#else
int color_main(int argc, char** argv)
#endif

{
	VideoCapture cap(0); //capture the video from web cam

	if ( !cap.isOpened() )  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	int yLowH = 22; //22
	int yHighH = 38; //38

	int yLowS = 200;  //160
	int yHighS = 255; //255

	int yLowV = 100; //60
	int yHighV = 255; //255
    
    int cLowH = 86;
    int cHighH = 106;
    
    int cLowS = 100;
    int cHighS = 255;
    
    int cLowV = 100;
    int cHighV = 255;

	while (true)
	{
		Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		//imgOriginal = imread( argv[1], 1 );
		//resize(imgOriginal,imgOriginal,Size(imgOriginal.cols/4,imgOriginal.rows/4));
		Mat imgHSV;

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded,imgThresholdedC,imgThresholdedY,edge;
        
        //Threshold the image
		inRange(imgHSV, Scalar(cLowH, cLowS, cLowV), Scalar(cHighH, cHighS, cHighV), imgThresholdedC);
        //inRange(imgHSV, Scalar(yLowH, yLowS, yLowV), Scalar(yHighH, yHighS, yHighV), imgThresholdedY);
        
        imgThresholded =   imgThresholdedC;

		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

		//morphological closing (fill small holes in the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );

		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		/// Convert it to gray
		//	Canny(  imgThresholded,edge, 150, 100,3 );
		//	imshow("Edge", edge); //show the original image

		vector<vector<Point> > contours;
		// find
		findContours( imgThresholded,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		drawContours( imgOriginal,contours,-1,Scalar(0,0,255),2);
		imshow("Original", imgOriginal); //show the original image
        
        //Canny
        Mat src = imgOriginal;
        Mat src1 = src.clone();
        Canny( src, src, 150, 100,3 );
        imshow("showcase Canny edge", src);
        Mat dst,gray;
        
        // dst has same size with src
        dst.create( src1.size(), src1.type() );
        
        // grey scale
        cvtColor( src1, gray, CV_BGR2GRAY );
        
        // 3*3 blur
        blur( gray, edge, Size(3,3) );
        
        // canny edge detection
        Canny( edge, edge, 3, 9,3 );
        
        //
        dst = Scalar::all(0);
        
        //
        src1.copyTo( dst, edge);
        
        //
        imshow("showcaseCanny edge2", dst);


		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break; 
		}	}
	return 0;

}

