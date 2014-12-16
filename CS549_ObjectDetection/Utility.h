/*
Description	:	Definitions and include files.
*/
#pragma once

/*
* Definitions
*/
#define MAIN
//#define COLOR_PRE

//#define SAMPLE_PEOPLE
//#define SAMPLE_TRIANGLE
//#define SAMPLE_MULTIPLESHAPES
//#define SAMPLE_IMAGE
//#define SAMPLE_COLOR



/*
* Include files
*/
#include<iostream>
#include<fstream>
#include<stdio.h>
#include <string.h>
#include<dirent.h>

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
//training
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;


/*Important: svmlight.h should be included after the namespce is defined, 
and should make distinction for svmlight::WORD to identify it to std::WORD*/


enum LOGTYPE  {LOG, ERROR, MESSAGE};
//log type strings should be consistant with LOGTYPE
static vector<string> logTypeStr = { "[LOG]: ", "[ERROR]: ", "[MESSAGE]: " };

class Utility
{
public:

	static void log(string logStr, int type = LOGTYPE::LOG)
	{
		cout << logTypeStr[type] << logStr << endl;
	}

	/* ----------------------------------------------------------------------- */
	/* Function    : Utility::split
	*
	* Description : split a string using a pattern
	*
	* Parameters  : string str		: the input string
	*					   string pattern: the split pattern
	*
	* Returns	     : vector<string> : a list of strings
	*/
	static vector<string> split(string str, string pattern)
	{
		std::string::size_type pos;
		std::vector<std::string> result;
		str += pattern; //add a pattern at the end
		int size = str.size();

		for (int i = 0; i < size; i++)
		{
			pos = str.find(pattern, i);
			if (pos < size)
			{
				std::string s = str.substr(i, pos - i);
				if (s != "")
				{
					result.push_back(s);
				}
				i = pos + pattern.size() - 1;
			}
		}
		return result;
	}

	static bool getFilesInDirectory(string dirStr, vector<string>& fileList, vector<string> validExt)
	{
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir(dirStr.c_str())) != NULL) {
			/* print all the files and directories within directory */
			while ((ent = readdir(dir)) != NULL) {
				vector<string> name = Utility::split(ent->d_name, ".");
				if (name.size() < 2)
					continue;
				for (int index = 0; index < validExt.size(); index++)
				{
					//turn ext to lowercase
					transform(name[1].begin(), name[1].end(), name[1].begin(), tolower);
					if (name[1] == validExt[index])
					{
						fileList.push_back(dirStr + ent->d_name);
					}
				}

			}
			Utility::log(to_string(fileList.size()) + " files got from foler " + dirStr);
			closedir(dir);
			return true;
		}
		else {
			/* could not open directory */
			perror("Could not open the directory.");
			return false;
		}
	}
};

