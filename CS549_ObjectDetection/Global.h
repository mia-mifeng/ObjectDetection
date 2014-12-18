//The file is for global variables and functions
#pragma once
#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include"Utility.h"

#define TRAINHOG_USEDSVM SVMLIGHT
//#define TRAINHOG_USEDSVM LIBSVM

#if TRAINHOG_USEDSVM == SVMLIGHT
#include "svmlight/svmlight.h"
#define TRAINHOG_SVM_TO_TRAIN SVMlight
#elif TRAINHOG_USEDSVM == LIBSVM
#include "libsvm/libsvm.h"
#define TRAINHOG_SVM_TO_TRAIN libSVM
#endif

/*
Global Variables
*/
//color recognition
const Vec3i blueTop = Vec3i(106, 255, 200);//HSV
const Vec3i blueBottom = Vec3i(86, 100, 100);//HSV
const int bluePercent = 16;
const Vec3i yellowTop = Vec3i(35, 255, 255);//29, 254, 255
const Vec3i yellowBottom = Vec3i(18, 0, 100); //23, 49, 149
const int yellowPercent = 2;

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);

//Directories for positive and negative training samples
string posSamplesDir = "Res\\sample\\positive\\";
string negSamplesDir = "Res\\sample\\negative\\";
vector<string> positiveTrainingImages;
vector<string> negativeTrainingImages;

// Set the file to write the features to
static string featuresFile = "Res\\genfiles\\features.dat";
static string svmModelFile = "Res\\genfiles\\svmlightmodel.dat";
static string descriptorVectorFile = "Res\\genfiles\\descriptorvector.dat";
static string svmParamsFile = "Res\\genfiles\\svmparam.txt";

vector<string> validExtensions = { "jpg", "png" };

string posTestDir = "Res\\image\\pos\\";
string negTestDir = "Res\\image\\neg\\";

vector<float> descriptorVector;
HOGDescriptor hog; // Use standard parameters here

struct SVMParam
{
	double hitThreshold;
};

SVMParam params;
/*
Global Functions declaration
*/

//imort samples
int importImageSamples(string posDir, vector<string>& posImgs,
	string negDir, vector<string>& negImgs, vector<string> validExt);
//sample files --> feature vector
int getSamplesToGenerateFeatures(vector<string>& posImgs, vector<string>& negImgs,
	HOGDescriptor hog, string featuresFile);
//calculate feature vector for each sample
static void calculateFeaturesFromInput(const string& imageFilename, 
	vector<float>& featureVector, HOGDescriptor& hog);
//SVM training
static void SVMTraining(string featuresFile, string svmModelFile, 
	vector<float>& descriptorVector);
//save/read params: the params can be loaded, instead of whole model, during detection
int saveSVMParamsToFile(string fileName);
int readSVMParamsFromFile(string fileName, SVMParam& params);
//save descriptor vector
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, 
	vector<unsigned int>& _vectorIndices, string fileName);
//read descriptor vector
int readDescriptVectorFromFile(vector<float>& descriptorVector, string fileName);
//detection
static void detectTestPN(const HOGDescriptor& hog, double hitThreshold,
	const vector<string>& posFileNames, const vector<string>& negFileNames);


/**
* Test the trained detector against the same training set to get an approximate idea of the detector.
* Warning: This does not allow any statement about detection quality, as the detector might be overfitting.
* Detector quality must be determined using an independent test set.
* @param hog
*/
static void detectTestPN(const HOGDescriptor& hog, double hitThreshold,
	const vector<string>& posFileNames, const vector<string>& negFileNames) {

	importImageSamples(posSamplesDir, positiveTrainingImages,
		negSamplesDir, negativeTrainingImages, validExtensions);

	unsigned int truePositives = 0;
	unsigned int trueNegatives = 0;
	unsigned int falsePositives = 0;
	unsigned int falseNegatives = 0;
	vector<Point> foundDetection;
	// Walk over positive training samples, generate images and detect
	for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
		Mat imageData = imread(*posTrainingIterator, 0);
		hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
		if (foundDetection.size() > 0) {
			++truePositives;
		}
		else {
			++falseNegatives;
			cout << "False negative: " << *posTrainingIterator << endl;
		}
		imageData.release();
	}
	// Walk over negative training samples, generate images and detect
	for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
		const Mat imageData = imread(*negTrainingIterator, 0);
		hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
		if (foundDetection.size() > 0) {
			falsePositives += 1;
			cout << "False positive: " << *negTrainingIterator << endl;
		}
		else {
			++trueNegatives;

		}
	}

	printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
	Utility::log("Accurate Rate: Positive: " + to_string(100 * truePositives / posFileNames.size()) + "%"
		+ " Negative: " + to_string(100 * trueNegatives / negFileNames.size()) + "%"
		+ " Overall: " + to_string(100 * (truePositives + trueNegatives) / (posFileNames.size() + negFileNames.size())) + "%");
}



/**
* This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
* @param imageFilename file path of the image file to read and calculate feature vector from
* @param descriptorVector the returned calculated feature vector<float> ,
*      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
* @param hog HOGDescriptor containin HOG settings
*/
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
	/** for imread flags from openCV documentation,
	* @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
	* @note If you get a compile-time error complaining about following line (esp. imread),
	* you either do not have a current openCV version (>2.0)
	* or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
	*/
	Mat imageData = imread(imageFilename, 0);
	if (imageData.empty()) {
		featureVector.clear();
		printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
		return;
	}
	// Check for mismatching dimensions
	if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
		featureVector.clear();
		printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
		return;
	}
	vector<Point> locations;
	hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
	imageData.release(); // Release the image again after features are extracted
}

int importImageSamples(string posDir, vector<string>& posImgs,
	string negDir, vector<string>& negImgs, vector<string> validExt)
{
	///*************Get the files to train****************/
	posImgs.clear();
	negImgs.clear();
	Utility::getFilesInDirectory(posDir, posImgs, validExt);
	Utility::getFilesInDirectory(negDir, negImgs, validExt);

	/// Retrieve the descriptor vectors from the samples
	unsigned long overallSamples = posImgs.size() + negImgs.size();

	// Make sure there are actually samples to train
	if (overallSamples == 0) {
		printf("No training sample files found, nothing to do!\n");
		return EXIT_SUCCESS;
	}
}

int getSamplesToGenerateFeatures(vector<string>& posImgs, vector<string>& negImgs,
	HOGDescriptor hog, string featuresFile)
{
	/// Retrieve the descriptor vectors from the samples
	unsigned long overallSamples = posImgs.size() + negImgs.size();
	/*************read the sample files****************/
	/**
	* Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
	* @NOTE: If you split these steps into separate steps:
	* 1. calculating features into memory (e.g. into a cv::Mat or vector< vector<float> >),
	* 2. saving features to file / directly inject from memory to machine learning algorithm,
	* the program may consume a considerable amount of main memory
	*/
	// @WARNING: This is really important, some libraries (e.g. ROS) 
	//seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail

	//pre-process
	setlocale(LC_ALL, "C"); // Do not use the system locale
	setlocale(LC_NUMERIC, "C");
	setlocale(LC_ALL, "POSIX");

	Utility::log("Reading files, generating HOG features and save them to file " + featuresFile);
	float percent;
	//open the feature file to write
	fstream File;
	File.open(featuresFile.c_str(), ios::out);
	if (File.good() && File.is_open()) {
		// Remove following line for libsvm which does not support comments
		// File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl;
		// Iterate over sample images
		for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
			vector<float> featureVector;
			// Get positive or negative sample image file path
			const string currentImageFile = (currentFile < posImgs.size() ? posImgs.at(currentFile) : negImgs.at(currentFile - posImgs.size()));
			// Output progress
			if ((currentFile + 1) % 10 == 0 || (currentFile + 1) == overallSamples) {
				percent = ((currentFile + 1) * 100 / overallSamples);
				printf("%5lu (%3.0f%%):\tFile '%s'\n", (currentFile + 1), percent, currentImageFile.c_str());
				fflush(stdout);
			}
			// Calculate feature vector from current image file
			calculateFeaturesFromInput(currentImageFile, featureVector, hog);
			if (!featureVector.empty()) {
				/* Put positive or negative sample class to file,
				* true=positive, false=negative,
				* and convert positive class to +1 and negative class to -1 for SVMlight
				*/
				File << ((currentFile < posImgs.size()) ? "+1" : "-1");
				// Save feature vector components
				for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
					File << " " << (feature + 1) << ":" << featureVector.at(feature);
				}
				File << endl;
			}
		}
		printf("\n");
		File.flush();
		File.close();
	}
	else {
		Utility::log("Error opening file " + featuresFile, LOGTYPE::ERROR);
		return EXIT_FAILURE;
	}

	/*************Get the files to train****************/
}
/**
* Saves the given descriptor vector to a file
* @param descriptorVector the descriptor vector to save
* @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
* @param fileName
* @TODO Use _vectorIndices to write correct indices
*/
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
	printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
	string separator = " "; // Use blank as default separator between single features
	fstream File;
	float percent;
	File.open(fileName.c_str(), ios::out);
	if (File.good() && File.is_open()) {
		printf("Saving %lu descriptor vector features:\t", descriptorVector.size());
		//storeCursor();
		for (int feature = 0; feature < descriptorVector.size(); ++feature) {
			if ((feature % 10 == 0) || (feature == (descriptorVector.size() - 1))) {
				percent = ((1 + feature) * 100 / descriptorVector.size());
				printf("%4u (%3.0f%%)", feature, percent);
				fflush(stdout);
				//resetCursor();
			}
			File << descriptorVector.at(feature) << separator;
		}
		printf("\n\n");
		File << endl;
		File.flush();
		File.close();
	}
}

int readDescriptVectorFromFile(vector<float>& descriptorVector, string fileName)
{
	Utility::log("Reading descriptor vector from file " + fileName);
	descriptorVector.clear();
	fstream File;
	float percent;
	File.open(fileName.c_str(), ios::in);
	if (!File.good() || !File.is_open())
	{
		Utility::log("Cannot open descriptor vector file " + fileName);
		return EXIT_FAILURE;
	}
	float featureElement;
	while (!File.eof())
	{
		File >> featureElement;
		descriptorVector.push_back(featureElement);
	}
	File.close();
	return EXIT_SUCCESS;
}





static void SVMTraining(string featuresFile, string svmModelFile, vector<float>& descriptorVector)
{
	/*************SVM gen model file****************/
	//SVM training
	// <editor-fold defaultstate="collapsed" desc="Pass features to machine learning algorithm">
	/// Read in and train the calculated feature vectors
	printf("Calling %s\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
	TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
	TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
	printf("Training done, saving model file!\n");
	TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);
	// </editor-fold>
	/*************SVM gen model file****************/


	/*************SVM gen descriptor vector file****************/
	// <editor-fold defaultstate="collapsed" desc="Generate single detecting feature vector from calculated SVM support vectors and SVM model">
	printf("Generating representative single HOG feature vector using svmlight!\n");
	vector<unsigned int> descriptorVectorIndices;
	// Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
	TRAINHOG_SVM_TO_TRAIN::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
	// And save the precious to file system
	saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
	// </editor-fold>
	/*************SVM gen descriptor vector file****************/
}

//should have model before doing this
int saveSVMParamsToFile(string fileName)
{
	fstream file;
	file.open(fileName, ios::out);
	float hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();
	if (!file.good() || !file.is_open())
	{
		file.close();
		return EXIT_FAILURE;
	}
	file << hitThreshold;
	file.close();
	return EXIT_SUCCESS;
}

int readSVMParamsFromFile(string fileName, SVMParam& params)
{
	fstream file;
	file.open(fileName, ios::in);
	
	if (!file.good() || !file.is_open())
	{
		file.close();
		return EXIT_FAILURE;
	}
	file >> params.hitThreshold;
	file.close();
	return EXIT_SUCCESS;
}

bool imageDetection( Mat& img, HOGDescriptor& hog)
{
	//HoG
	vector<Rect> found, filteredFound;
	if (img.cols <= 480 && img.rows <= 480)
	{
		hog.detectMultiScale(img, found, params.hitThreshold, Size(16, 16), trainingPadding, 1.1);
	}
	else
	{
		hog.detectMultiScale(img, found, params.hitThreshold, Size(8, 8), trainingPadding, 4);
	}
	

	/*if (img.cols * img.rows > 2 ^ 17)
	{
		if (img.cols>img.rows)
		{
			resize(img, img, Size(320, 320*img.rows/img.cols));
		}
		else
		{
			resize(img, img, Size(img.cols * 320/img.rows, 320));
		}
	}*/
	//debug log
	//if (found.size())
	//	cout << "Found number: " << found.size() << endl;

	//color verify
	int maxPercent = 0; int maxIndex = 0;
	Mat imgHSV, imgThresholded;
	cvtColor(img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
	
	//Threshold the image
	inRange(imgHSV, blueBottom, blueTop, imgThresholded);

	//morphological opening (remove small objects from the foreground)
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//morphological closing (fill small holes in the foreground)
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

	vector<vector<Point> > contours;
	// find
	findContours(imgThresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(img, contours, -1, Scalar(255, 0, 0), 1);

	Mat yellowThres;
	//Threshold the image
	inRange(imgHSV, yellowBottom, yellowTop, yellowThres);

	//morphological opening (remove small objects from the foreground)
	erode(yellowThres, yellowThres, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(yellowThres, yellowThres, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//morphological closing (fill small holes in the foreground)
	dilate(yellowThres, yellowThres, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(yellowThres, yellowThres, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
	
	contours.clear();
	// find
	findContours(yellowThres, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(img, contours, -1, Scalar(0, 255, 255), 1);

	for (int index = 0; index < (found.size()<1?found.size():1); index++) //no more than 1
	//for (int index = 0; index < found.size() ; index++)
	{
		Rect rect = found[index];
		
		int green = 50;
		if (index == 0)
			green = 255;

		//verification
		int percent = 0;
		for (int i = (rect.tl().y<0 ? 0 : rect.tl().y);
			i < (rect.br().y > img.rows ? img.rows : rect.br().y); i++)
		{
			for (int j = (rect.tl().x<0?0:rect.tl().x); 
				j < (rect.br().x > img.cols ? img.cols : rect.br().x); j++)
			{
				percent += imgThresholded.at<uchar>(i, j) % 254;
			}
		}
		percent = percent * 100 / rect.area();
		
		if (percent > bluePercent)
		{
			filteredFound.push_back(found[index]);
			if (percent > maxPercent)
			{
				maxPercent = percent;
				maxIndex = filteredFound.size()-1;
			}
		}

	}

	if (filteredFound.size())
	{
		Rect rect = filteredFound[maxIndex];
		//verification
		int percent = 0;
		for (int i = (rect.tl().y<0 ? 0 : rect.tl().y);
			i < (rect.br().y > img.rows ? img.rows : rect.br().y); i++)
		{
			for (int j = (rect.tl().x<0 ? 0 : rect.tl().x);
				j < (rect.br().x > img.cols ? img.cols : rect.br().x); j++)
			{
				percent += yellowThres.at<uchar>(i, j) % 254;
			}
		}
		percent = percent * 100 / rect.area();
		if (percent > yellowPercent)
		{
			rectangle(img, rect.tl(), rect.br(), cv::Scalar(0, 255, 0), 2);
		}
	}
	if (img.cols <= 480 && img.rows <= 480)
	{
		resize(img, img, Size(img.cols * 2, img.rows * 2));
	}
	
	imshow("Object Detection", img);
	img.release();
	
	return found.size()?true:false;
}

void showInstructions()
{
	cout << endl;
	cout << "****************Command Instructions:*************** " << endl;
	cout << "f: feature generation" << endl;
	cout << "t: training with SVM" << endl;
	cout << "s: detect training samples, console output" << endl;
	cout << "p: parameter adjustment, console output" << endl;
	cout << "c: detect images with console output" << endl;
	cout << "i: detect image slides" << endl;
	cout << "v: detect video" << endl;
	cout << "r: detect real time webcam" << endl;
	cout << "q: quit" << endl;
	cout << "Command>>";
}

#endif

