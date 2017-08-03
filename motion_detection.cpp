
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
using namespace std;
using namespace cv;

/*Global variable declarations*/
Rect boundRect;
Point center,prevcenter;
RNG rng(12345);
int countcars =0,counttrucks=0;
Mat frame,current;
string kpfname;
Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );


/*Function to draw contour and get minimum bounding rectangle of the object region*/
void getRect(Mat framemask)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours( framemask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	Mat drawing = Mat::zeros( framemask.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
	}
	boundRect.x=0;
	boundRect.y=0;
	boundRect.height=0;
	boundRect.width=0;
	double maxArea=20000; //Minumu area if object is a car would be greater than 20000
	vector<vector<Point> > contours_poly( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{ 
		double area = contourArea(contours[i]);
		if(area>maxArea)
		{
			maxArea = area;
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect = boundingRect(Mat(contours_poly[i])); //Getting boundRect
		}
	}

}

/* Function findcolor: Gets maximum blue,gree,red intensities and converts the image to HSV channel and prints color based on Hue*/
void findcolor(Mat currentframe,int veh)
{
	int blue=0,green=0,red =0,i=0,j=0,cnt=0;
	int hue=0,sat=0,brightness=0;
	string vehicle;
	
	//To pinrt car or truck and its corresponding count
	if(veh==1)
	{
		vehicle="car";
		cnt=countcars;
	}
	else
	{
		vehicle = "truck";
		cnt=counttrucks;
	}
	//summing up all the intensities
	for(i=0;i<=currentframe.rows;i++)
	{
		for(j=0;j<=currentframe.cols;j++)
		{
			blue += currentframe.at<Vec3b>(i,j)[0];
			green += currentframe.at<Vec3b>(i,j)[1];
			red += currentframe.at<Vec3b>(i,j)[2];
		}
	}
	
	if(red==blue && blue == green)
		cout<<vehicle<<":"<<cnt<<":white";
	else
	{
		cvtColor( currentframe, currentframe, CV_BGR2HSV );
	
		for(i=0;i<=currentframe.rows;i++)
		{
			for(j=0;j<=currentframe.cols;j++)
			{
				hue += currentframe.at<Vec3b>(i,j)[0];
				sat += currentframe.at<Vec3b>(i,j)[1];
				brightness += currentframe.at<Vec3b>(i,j)[2];
			}
		}
		hue = hue/(currentframe.rows*currentframe.cols);
	
		sat = sat/(currentframe.rows*currentframe.cols);
		brightness = brightness/(currentframe.rows*currentframe.cols);
		if(sat == 0 && brightness!=0)
			cout<<vehicle<<":"<<cnt<<":Gray";
		else if(brightness ==0)
			cout<<vehicle<<":"<<cnt<<":Black";
		else
		{
		if(hue>=160 && hue <=179)
			cout<<vehicle<<":"<<cnt<<":Red";
		else if(hue>=130 && hue <=159)
			cout<<vehicle<<":"<<cnt<<":Violet";
		else if(hue>=75 && hue <130)
			cout<<vehicle<<":"<<cnt<<":Blue";
		else if(hue>=38 && hue <75)
			cout<<vehicle<<":"<<cnt<<":Green";
		else if(hue>=22 && hue <38)
			cout<<vehicle<<":"<<cnt<<":Yellow";
		else if(hue>=0 && hue <22)
			cout<<vehicle<<":"<<cnt<<":Orange";
		}
		
		//cout<<"hue:"<<hue;
		//cout<<"sat:"<<sat;
		//cout<<"brightness:"<<brightness<<"\n";
	}	
}
int main()
{
	string file_name,video_out;
	Mat res,frameMOG,background,framegray,backgroundgray,prevframe;
	int w=0,h=0,x=0,y=0,i=0,j=0,fps,fourcc=0;

	int enteredregion =0;
	int index = 0;
	int currArea,prevArea =0;
	int cardetected,prevframecar;
	
	//Reading Filename from user
	cout<<"Please enter file name without spaces: ";
	cin>>file_name;
	video_out = "output.mkv";
	
	//Creating VideoCapture object for opening file
	VideoCapture cap(file_name);
	if(!cap.isOpened())
	{
		cout<<"Could not open file";
		return -1;
	}
	//Mog object creation for background subtraction
	Ptr<BackgroundSubtractor> ptrMOG;
	ptrMOG = new BackgroundSubtractorMOG(); 
	cap.read(background);
	

	GaussianBlur(background,background,Size(9,9),0,0,BORDER_DEFAULT);
	cvtColor(background, backgroundgray, CV_BGR2GRAY );
	medianBlur(backgroundgray,backgroundgray,9);
	
	w=int(cap.get(CV_CAP_PROP_FRAME_WIDTH));
	h=int(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	fps=int(cap.get(CV_CAP_PROP_FPS));
	VideoWriter writer(video_out,CV_FOURCC('F','M','P','4'),fps,Size(w,h));
	if(!writer.isOpened())
	{
		cout<<"Could not open file";
		return -1;
	}
	
	//Reading a frame and setting intesity of ROI to 255 
	while(cap.read(frame))  
	{		
			//Applying following operations in sequnce to each frame
			//1. GaussianBlur
			//2.GrayScale conversion
			//3.MedianBlur
			//4.MOG Operator for background subtraction
			//5.thresholding with mean+20 intensity
			//6.Canny edge detection
			//7.MorphologyEx to close holes
			//8.Finding contours
			//9. Getting Bounding box
			
		frame.copyTo(current);
		GaussianBlur(current,current,Size(9,9),0,0,BORDER_DEFAULT);
		cvtColor( current, framegray, CV_BGR2GRAY );
		medianBlur(framegray,framegray,9);
		
		ptrMOG->operator()(framegray, frameMOG);
		//absdiff(backgroundgray,framegray,frameMOG);
		
		Scalar meanIntensity = mean(frameMOG,noArray());
		double threslimit = meanIntensity.val[0];
		threshold(frameMOG,frameMOG,threslimit+20,255,CV_THRESH_OTSU);
		//threshold(frameMOG,frameMOG,threslimit+10,255,CV_THRESH_BINARY);
		
		Canny( frameMOG, frameMOG, threslimit+20, 100, 3 );
		morphologyEx(frameMOG,frameMOG,MORPH_CLOSE,Mat(),Point(-1,-1),20);
		//floodFill(frameMOG,Point(),Scalar(),0,Scalar(),Scalar(),4);
		
		getRect(frameMOG);
		
		if(boundRect.width==0||boundRect.height==0)
		{
			frame.copyTo(background);
			GaussianBlur(background,background,Size(9,9),0,0,BORDER_DEFAULT);
			cvtColor( background, backgroundgray, CV_BGR2GRAY );
			medianBlur(backgroundgray,backgroundgray,9);
			//enteredregion==0;
		}
		else
		{
			frame.copyTo(current);
			rectangle( frame, boundRect.tl(), boundRect.br(),color, 4, 8, 0 );
			center.x = boundRect.x+( boundRect.width/2);
			center.y = boundRect.y+( boundRect.height/2);
			circle(frame,center,4,color,-1,8,0);
			currArea = boundRect.width*boundRect.height;
			//Checking if center is in the band of interest, once entered, entredregion is markd 1 to avoid counting same vehicle again
			if(center.y>150&&center.y<200 && enteredregion==0)
			{
				enteredregion=1;
				stringstream ss;
				//Truck area would be more than 60% of the video
				if(currArea > (frame.rows*frame.cols*0.6))
				{
					counttrucks++;
					current(boundRect);
					findcolor(current,2);
					kpfname = "truck_";
					ss<<counttrucks;
				}
				else
				{
					countcars++;
					current(boundRect);
					findcolor(current,1);
					kpfname = "car_";
					ss<<countcars;	
				}
				kpfname = kpfname+ss.str()+".jpg";
				ss.clear();
				imwrite(kpfname,frame);
				cardetected=1;
			}
			else if(center.y<150 || center.y>200)
			{
				enteredregion=0;
			}
			prevArea = currArea;
		}
		//color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		rectangle( frame, Point(0,150), Point(1280,200),color, 4, 8, 0 );
		string videotext = "CARS=";
		stringstream ss;
		ss<<countcars;
		videotext=videotext+ss.str();
		ss.clear();
		putText(frame, videotext, cv::Point(700, 50), FONT_HERSHEY_SIMPLEX, 1 , Scalar(40,50,60));
		
		videotext = "TRUCKS=";
		stringstream st;
		st<<counttrucks;
		videotext=videotext+st.str();
		st.clear();
		putText(frame, videotext, cv::Point(1100, 50), FONT_HERSHEY_SIMPLEX, 1 , Scalar(40,50,60));
		
		writer.write(frame);
		/*if(cardetected)
		{
			int i=15;
			while(i)
			{
			cap.read(frame);
			i--;
			}
			cardetected=0;
		}	*/
		
	}
	writer.release();
	cap.release();
	
	return 0;
}