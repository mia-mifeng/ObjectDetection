/*
Description	:	People detection based on HOG feature and SVM classifier, 
						with video capture and green rectangle display.
Source			:	http://www.magicandlove.com/blog/2011/08/26/people-detection-in-opencv-again/
*/
#include"Utility.h"

#ifdef SAMPLE_PEOPLE
int main(int argc, char** argv)
#else
int sample_people_main(int argc, char** argv)
#endif
{
	VideoCapture cap(CV_CAP_ANY);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!cap.isOpened())
		return -1;

	Mat img;
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	namedWindow("video capture", CV_WINDOW_AUTOSIZE);
	while (true)
	{
		cap >> img;
		if (!img.data)
			continue;

		vector<Rect> found, found_filtered;
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);

		size_t i, j;
		for (i = 0; i<found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j<found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		for (i = 0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.06);
			r.height = cvRound(r.height*0.9);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
		imshow("video capture", img);
		if (waitKey(20) >= 0)
			break;
	}
	return 0;
}

