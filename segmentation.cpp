#pragma package <opencv>
#include <iostream>
#include <unistd.h>
#include <time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>
#include<vector>
#include "hog.cpp"
using namespace cv;
using namespace std;
int MINHEIGHT,MIN_CONTOUR_AREA=1000;
int sthresh=0;  //threshold for interword spaces vs intraword spaces
class ContourWithData {
public:
	// member variables ///////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point> ptContour;			// contour
	cv::Rect boundingRect;						// bounding rect for contour
	float fltArea;								// area of contour

	///////////////////////////////////////////////////////////////////////////////////////////////
	bool checkIfContourIsValid() {									// obviously in a production grade program
		if (fltArea < MIN_CONTOUR_AREA) return false;				// we would have a much more robust function for 
		return true;												// identifying if a contour is valid !!
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	static bool sortByBoundingRectPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {		// this function allows us to sort
		if(abs(cwdLeft.boundingRect.y - cwdRight.boundingRect.y)>MINHEIGHT*1.3)
		    return(cwdLeft.boundingRect.y < cwdRight.boundingRect.y);													// the contours from left to right
		 return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
	}

};

vector<ContourWithData> allContoursWithData;
vector<ContourWithData> validContoursWithData;
bool equal(int h[],int h2[],int l)  {
    for(int i=0;i<l;i++)    {
        if(h[i]!=h2[i]) return false;
    }
    return true;
}
int cluster(int c[],int l)  {
    int a=c[0],b=c[l/2];
    int *h=new int[l];
    int *h2=new int[l];
    do {
        for(int i=0;i<l;i++)    {
            h2[i]=h[i];
            h[i]=(abs(c[i]-a) > abs(c[i]-b)) ? 1 : 0;
        }
        a=0;b=0;
        int la=0,lb=0;
        for(int i=0;i<l;i++)    {
            if(h[i]==0) {
                a+=c[i];
                la++;
            }
            else if(h[i]==1)    {
                b+=c[i];
                lb++;
            }
        }
        a=a/la;
        b=b/lb;
    }while(!equal(h,h2,l) );
    int mn=0,mx=0;
    for(int i=0;i<l;i++)    {
        //printf("%d %d\n",c[i],h[i]);
        if(h[i]==0) {
            mx=(c[i]>mx)?c[i]:mx;
        }
        else if(h[i]==1)    {
            mn=(mn==0 || c[i]<mn)?c[i]:mn;
        }
    }
    //printf("%d %d\n",mn, mx);
    return (mx + 1);
}
char* segword(Mat m, Mat org, Rect r,char *mode)   {

    int *a=new int[r.width];	
    int f=0,c=0;
	int cc=0,kk=0;
    int *b=new int[r.width];	    //Array to hash no of spaces in line, i.e. ith index=1 denotes line has i continuous spaces.
    int *d=new int[r.width];        //Array to store no of spaces starting at ith index.
	for (int i = 0; i < r.width; i++) 
	{
		a[i]=0;
		b[i]=0;
    }
    
	for (int i = 0; i < r.width; i++) {
	    d[i]=0;
		for(int j=0;j<r.height;j++)	{
			Scalar myColor = m.at<uchar>(Point(i+r.x,j+r.y));
			int pixelValue = myColor.val[0]/255;
			a[i]+=pixelValue;
		}
		if(a[i]!=0) {
		    if(f==0)    {
		        d[kk]=c;
		    }
		    kk=i;
		    f=1;
		    c=0;
		}
		else    {
		    if(f==0)    c++;
		    else if(f==1)  { kk=i; c=1; f=0;}
		}
	}
	if(f==0 && a[r.width-1]==0)    {
	    d[kk]=c;
	}
    for(int i=d[0];i<kk;i++)  {
        if(d[i]!=0) {
	        if(b[d[i]]==0) cc++;
            b[d[i]]++;
        }
    }
	int *c2=new int[cc];
	int in2=0;
	for (int i = 0; i < r.width; i++)   
	{
        if(b[i]!=0) {
            c2[in2++]=i;
            //if(in2==2 && 10*i<r.width)  { b[10*i]=1;  cc++;}        
            //printf("%d\n",i);
        }
    }
    char *chr=new char[100000];
    int indx=0;
    
    Mat drawing = Mat::zeros( m.size(), CV_8UC1 );
    org.copyTo(drawing);
    if(mode[0]=='1') {
        Rect rec(d[0],r.y,kk-d[0]+1,r.height);
        rectangle(drawing,rec, Scalar(0,0,255), 1);
        hog_train(m,org,rec);
    }
    else    {
	    int wThresh=cluster(c2,cc);
	    if(sthresh==0)
	        sthresh=wThresh;
	    if(abs(sthresh-wThresh)>sthresh)
	        sthresh=min(sthresh,wThresh);
	    else
	        sthresh=max(sthresh,wThresh);
	    //printf("hell %d %d\n",wThresh,sthresh);
	    int left=d[0],len=0;
	    for(int i=d[0];i<r.width;i++)  {
	        if(a[i]==0)   {
	            if(d[i]>sthresh)    {
	                Rect rec(left-1,r.y,i-left+1,r.height);
	                rectangle(drawing,rec, Scalar(0,0,255), 1);
	                if(mode[0]=='2') hog_train2(m,org,rec);
	                else {  
	                    hog(m,org,rec,chr,&indx);
	                    chr[indx++]=' ';
	                }
	                i+=d[i];
	                left=i;
	            }
	        }
	    }
	    chr[indx++]='\0';
	    printf("%s\n",chr);
	    FILE *fp=fopen("out.txt","a+");
	    fprintf(fp,"%s\n",chr);
	    fclose(fp);
	}
    namedWindow("word", CV_WINDOW_AUTOSIZE); 
    imshow("word",drawing);
	//waitKey(0);
	return chr;
	
}

Mat deskew(Mat src)    {
    
    
    return src;
}
void segment(Mat m,Mat org,char *mode)
{
    MINHEIGHT=m.rows;
    int thresh=0;
    int *a=new int[m.rows];	
    //Mat rr=deskew(m);
    //waitKey(0);
	for (int i = 0; i < m.rows; i++) 
	{
		a[i]=0;
		for(int j=0;j<m.cols;j++)	{
			Scalar myColor = m.at<uchar>(i,j);
			int pixelValue = myColor.val[0]/255;
			a[i]+=pixelValue;
		}
	}
	int flag=1,strt=0,i;
    //Mat drawing = Mat::zeros( m.size(), CV_8UC1 );
    //org.copyTo(drawing);
	for(i=0;i<m.rows;i++) {
	    if(a[i]>thresh) flag=2;
	    else    {
	        if(flag==2) {
	            Rect rec(0,strt,m.cols,i-strt+1);
	            //rectangle(drawing,rec, Scalar(0,0,255), 1);
	            segword(m,org,rec,mode);
    	        flag=1; strt=i;
	        }
	        else    {
	            flag=1; strt=i;
	        }
	    }
	    
		//namedWindow("line", CV_WINDOW_AUTOSIZE); 
		//imshow("line",drawing);
	}
	if(strt!=i && flag==2) {
        Rect rec(0,strt,m.cols,i-strt-1);
        //rectangle(drawing,rec, Scalar(0,0,255), 1);
        segword(m,org,rec,mode);
        flag=1; strt=i;
    }
    //Mat tmp;
    //morphologyEx(m,m,MORPH_OPEN,tmp,Point(2,2));
	/*Mat m2 = m.clone(); 
    vector<vector<Point> > ptContours;		// declare contours vector
    vector<Vec4i> v4iHierarchy; 
    findContours(m2,			// input image, make sure to use a copy since the function will modify this image in the course of finding contours
		ptContours,					// output contours
		v4iHierarchy,					// output hierarchy
		RETR_EXTERNAL,				// retrieve the outermost contours only
        CHAIN_APPROX_SIMPLE);
         /// Draw contours
    
    waitKey(0);
    for (int i = 0; i < ptContours.size(); i++) {
        ContourWithData contourWithData;												// instantiate a contour with data object
	    contourWithData.ptContour = ptContours[i];										// assign contour to contour with data
        Rect br = cv::boundingRect(contourWithData.ptContour);		                    // get the bounding rect
        contourWithData.boundingRect=br;
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);			// calculate the contour area
        allContoursWithData.push_back(contourWithData);									// add contour with data object to list of all contours with data
    }
    for (int i = 0; i < allContoursWithData.size(); i++) {					// for all contours
		if (allContoursWithData[i].checkIfContourIsValid()) {			// check if valid	
		    Rect br = cv::boundingRect(allContoursWithData[i].ptContour);		// get the bounding rect
            if(MINHEIGHT>br.height)
                MINHEIGHT=br.height;
			validContoursWithData.push_back(allContoursWithData[i]);		// if so, append to valid contour list
		}
	}
   sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectPosition);

 Mat drawing = Mat::zeros( m.size(), CV_8UC1 );
 
  for( int i = 0; i< ptContours.size(); i++ )
     {
       Scalar color = Scalar( 255,255,255 );
       drawContours( drawing, ptContours, i, color, 2, 8, v4iHierarchy, 0, Point() );
     }
     imwrite("contours.png",drawing);
     imshow("imgContours",drawing);
    waitKey(0);
   for (vector<ContourWithData>::iterator it = validContoursWithData.begin() ; it != validContoursWithData.end(); ++it) {
        hog(tm,org,(*it).boundingRect);
        rectangle(org, (*it).boundingRect, Scalar(0, 0, 0), 0.2);
        
   }   namedWindow("Org", 0); 
    resizeWindow("Org", 1360,768);
        imshow("Org", org);
        //imwrite("words.png",org);	
        waitKey(0);*/
}

