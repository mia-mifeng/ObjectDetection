/*
Description	:	The main function of the project. Object detection, yay.
*/

//include"Utility.h"
#include"Global.h"


//#define GENFEATUREG
//#define TRAINING
//#define DETECTING

void generateFeature()
{
	//image import
	importImageSamples(posSamplesDir, positiveTrainingImages,
		negSamplesDir, negativeTrainingImages, validExtensions);

	//generate features
	getSamplesToGenerateFeatures(
		positiveTrainingImages,
		negativeTrainingImages,
		hog, featuresFile);
}

void buttonGenFeature(int state, void* userdata)
{
	//generateFeature();
}

void close()
{
	cvDestroyAllWindows();
}

void detectPrep()
{
	Utility::log("loading descriptor vector from file " + to_string(descriptorVector.size()));
	readDescriptVectorFromFile(descriptorVector, descriptorVectorFile);

	// Detector detection tolerance threshold
	params = SVMParam();
	readSVMParamsFromFile(svmParamsFile, params);
	//const double hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();
	Utility::log("Training Threshold: " + to_string(params.hitThreshold));

	// Set our custom detecting vector
	Utility::log("Set detecting vector");
	hog.setSVMDetector(descriptorVector);
}




void init()
{
	//initialization
	hog.winSize = Size(64, 128); // Default training images size as used in paper
	showInstructions();
	//create a window called "Control Panel"
	//namedWindow("Control Panel", CV_WINDOW_AUTOSIZE); 

	//cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	//cvCreateButton("Feature Generation", buttonGenFeature, NULL, CV_PUSH_BUTTON);
	//cv::createButton("Generate Feature", buttonGenFeature, NULL,CV_PUSH_BUTTON,0);
	//cvAddText()
}


#ifdef MAIN
int main(int argc, char** argv)
#else
int main_main(int argc, char** argv)
#endif
{
	init();

	
	string cmd;
	while (cin >> cmd)
	{
		
		if (cmd == "f") //f: feature generation
		{
			//image samples --> feature vectors
			generateFeature();
		}
		else if (cmd == "t") //t: training with SVM
		{
			//feature vectors --> descriptor vector
			SVMTraining(featuresFile, svmModelFile, descriptorVector);
			saveSVMParamsToFile(svmParamsFile);
		}
		else if (cmd == "s") //s: detect training samples, console output
		{
			detectPrep();
			detectTestPN(hog, params.hitThreshold, positiveTrainingImages, negativeTrainingImages);
		}
		else if (cmd.size() && cmd[0] == 'p') //s: detect training samples, console output
		{
			detectPrep();
			string sThres;
			if (cin >> sThres)
			{
				double thres = atof(sThres.c_str());
				Utility::log("Testing theshold=" + to_string(thres));
				detectTestPN(hog, thres, positiveTrainingImages, negativeTrainingImages);
			}
		}
		else if (cmd == "c") //c: detect images with console output
		{
			detectPrep();
			vector<string> posTestFiles, negTestFiles;
			Utility::getFilesInDirectory(posTestDir, posTestFiles, validExtensions);
			Utility::getFilesInDirectory(negTestDir, negTestFiles, validExtensions);
			detectTestPN(hog, params.hitThreshold, posTestFiles, negTestFiles);
		}
		else if (cmd == "i") //c: detect images with console output
		{
			detectPrep();
			vector<string> posTestFiles, negTestFiles;
			Utility::getFilesInDirectory("Res\\image\\pos\\", posTestFiles, validExtensions);
			Utility::getFilesInDirectory("Res\\image\\neg\\", negTestFiles, validExtensions);
			//namedWindow("Detecting Images", CV_WINDOW_AUTOSIZE);
			Mat img = imread(posTestFiles[0]);
			int posIndex = 0;
			int negIndex = 0;
			img = imread(posTestFiles[posIndex]);
			imageDetection(img, hog);
			
			int key = -1;
			while ( key != 113)
			{
				key = waitKey(30);
				switch (key)
				{
				case 112: //p: positive forward
					posIndex = (posIndex+1) % posTestFiles.size();
					img = imread(posTestFiles[posIndex]);
					imageDetection(img, hog);
					break;
				case 80: //P: positive back
					posIndex = (posIndex - 1) % posTestFiles.size();
					img = imread(posTestFiles[posIndex]);
					imageDetection(img, hog);
					break;
				case 110: //n: negative forward
					negIndex = (negIndex + 1) % negTestFiles.size();
					img = imread(negTestFiles[negIndex]);
					imageDetection(img, hog);
					break;
				case 78: //N: negative back
					negIndex = (negIndex - 1) % negTestFiles.size();
					img = imread(negTestFiles[negIndex]);
					imageDetection(img, hog);
					break;
				case 29: // left arrow
					break;
				case 30: // up arrow
					break;
				case 31: // down arrow
					break;
				case 113://q
					img.release();
					close();
					Utility::log("Exit image detection!");
					break;
				}
			}
		
		}
		else if (cmd == "v") //video
		{
			//reshape?
		}
		else if (cmd == "r") //real time
		{
			detectPrep();
			VideoCapture cap(CV_CAP_ANY);
			cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
			cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
			if (!cap.isOpened())
				return EXIT_FAILURE;

			Mat img;

			//key event
			int key = -1;
			while (key != 113)
			{
				cap >> img;
				//imshow("Real-time Detection YAY!!!", img);
				imageDetection(img, hog);

				//key
				key = waitKey(30);
				switch (key)
				{
				case 113: //q: quit
					img.release();
					close();
					Utility::log("Exit video detection!");
					break;
				default:
					break;
				}

			}
		}
		else if (cmd == "q") //quit
		{
			//close();
			return EXIT_SUCCESS;
		}

		showInstructions();
	}

	while (true)
	{
		int key = waitKey(30);
		switch (key)
		{
		case 28: // right arrow
			//detect training data
			detectPrep();
			detectTestPN(hog, params.hitThreshold, positiveTrainingImages, negativeTrainingImages);
			break;
		case 29: // left arrow
			break;
		case 30: // up arrow
			break;
		case 31: // down arrow
			break;
		case 27://esc
			close();
			Utility::log("Exit!");
			return EXIT_SUCCESS;
			//case 

		}
	}

#ifdef GENFEATUREG
	//image samples --> feature vectors
	generateFeature();
#endif

#ifdef TRAINING
	//feature vectors --> descriptor vector
	SVMTraining(featuresFile, svmModelFile, descriptorVector);
	saveSVMParamsToFile(svmParamsFile);
#endif
	
#ifdef DETECTING
	
	detectPrep();
	vector<string> posTestFiles, negTestFiles;
	Utility::getFilesInDirectory("Res\\image\\pos\\", posTestFiles, validExtensions);
	Utility::getFilesInDirectory("Res\\image\\neg\\", negTestFiles, validExtensions);
	detectTestPN(hog, params.hitThreshold, positiveTrainingImages, negativeTrainingImages);
	//detectTestPN(hog, hitThreshold, posTestFiles, negTestFiles);
#endif
	
	return 0;
}


