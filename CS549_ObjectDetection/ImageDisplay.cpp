/*
Description	:	Simple image display
*/

#include"Utility.h"

#ifdef SAMPLE_IMAGE
int main()
#else
int sample_image_main()
#endif
{
	Mat image = cvLoadImage("Res\\image\\Capture.PNG"); //notice the path of the image file
	Mat HSV;
	cvtColor(image, HSV, CV_BGR2HSV);
	Point center = Point(150,280);
	Vec3b hsv = HSV.at<Vec3b>(center);
	Vec3i max = hsv;
	Vec3i min = hsv;
	
	int b = 5;
	for (int i = -b; i < b; i++)
	{
		for (int j = -b; j < b; j++)
		{
			hsv = HSV.at<Vec3b>(center + Point(i, j));
			cout << "HSV" << (int)hsv.val[0] << " " << (int)hsv.val[1] << " " << (int)hsv.val[2] << endl;
			for (int t = 0; t < 3; t++)
			{
				if (hsv.val[t] < min[t])
				{
					min[t] = hsv.val[t];
				}
				if (hsv.val[t]>max[t])
				{
					max[t] = hsv.val[t];
				}
			}
		}
	}
	rectangle(image, center - Point(b, b), center + Point(b, b), cv::Scalar(0, 255, 0), 2);
	cout << "Max: " << max << endl;
	cout << "Min: " << min << endl;
	imshow("Test", image);

	cvWaitKey(0);
	image.release();
	//cvReleaseImage(&image);
	//cvDestroyWindow("Hello World");
	return 0;
}