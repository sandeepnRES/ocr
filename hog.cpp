#pragma package <opencv>
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <time.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>
#include<vector>
#define PI 3.14159265

using namespace cv;
using namespace std;
const int rw=32,rh=32;

Mat resize2(Mat m)   {
      int h=m.rows;
      int w=m.cols;
      Mat tmp = Mat::zeros( rh,rw, m.type() );
      Rect roi;
      if(w<h) {
            roi.height=rh;
            roi.width=(w*rh)/h;
            roi.x=(rw-roi.width)/2;
      }  
      else  {
            roi.width=rw;
            roi.height=(h*rw)/w;
            roi.y=(rh-roi.height)/2;
      }
      //printf("%d %d %d %d %d %d\n",roi.x,roi.y,roi.width,roi.height,tmp.cols,tmp.rows);
      resize(m,tmp(roi),roi.size());
      return tmp;
}
void normalize2(double *res,int h[][4][9]){
    double val=0.0;
    int k=0;
    for(int i=0;i<7;i++)  {
        for(int j=0;j<3;j++)  {
            //Calculating norm denominator
            val=0.0;
            for(int n=0;n<9;n++)    {
                val+=(h[i][j][n]*h[i][j][n]);
            }
            for(int n=0;n<9;n++)    {
                val+=(h[i+1][j][n]*h[i+1][j][n]);
            }
            for(int n=0;n<9;n++)    {
                val+=(h[i][j+1][n]*h[i][j+1][n]);
            }
            for(int n=0;n<9;n++)    {
                val+=(h[i+1][j+1][n]*h[i+1][j+1][n]);
            }
            val=sqrt(val);
            
            //Normalizing 4 9x1 Vectors
            for(int n=0;n<9;n++)    {
                res[k++]+=((double) h[i][j][n])/val;
            }
            for(int n=0;n<9;n++)    {
                res[k++]+=((double) h[i+1][j][n])/val;
            }
            for(int n=0;n<9;n++)    {
                res[k++]+=((double) h[i][j+1][n])/val;
            }
            for(int n=0;n<9;n++)    {
                res[k++]+=((double) h[i+1][j+1][n])/val;
            }
        }
        
    }
    printf("k=%d\n",k);
}
void train(Mat m,int i)   {
}
void recognize(Mat m)   {
    //Normalizing size of each character
    Mat img=resize2(m);
    HOGDescriptor hog;
    vector<float> featV;
    vector<Point>locs;
    hog.compute (img,featV,Size(8,8),Size(0,0),locs);
    //Computing vertical and horizontal gradient in Polar coordinates
    /*img.convertTo(img, CV_32F, 1/255.0);
    Mat gx, gy; 
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);
    Mat mag, angle; 
    cartToPolar(gx, gy, mag, angle, 1); 
    imshow("1",gx);
    imshow("2",gy);
    //Calculating histograms of 8x8 patches in 9 bins.
    int h1[8][4][9];
    int i1=0,j1=0;
    for(int i=0;i<64;i+=8)  {
        for(int j=0;j<32;j+=8)  {
            for(int l=0;l<8;l++)    {
                for(int m=0;m<8;m++)    {
                    //int vx=gx.at<uchar>(Point(i+l,j+m));
                    //int vy=gy.at<uchar>(Point(i+l,j+m));
                    //int vmag=mags[i+l][j+m];
                   // int ang=angs[i+l][j+m];
                    double gxv=(double) gx.at<uchar>(Point(i,j));
                    double gyv=(double) gy.at<uchar>(Point(i,j));
                    int vmag=sqrt(gxv*gxv+gyv*gyv);
                    double angss=atan2(gyv,gxv)*180/PI;;
                    if(angss<0) angss=180+angss;
                    int ang=(int) angss;
                    
                    printf("%d %d\n",vmag,ang);
                    int bno=(ang%180)/20;
                    int bdiff=ang%20;
                    if(bdiff==0)    {
                        h1[i1][j1][bno]+=vmag;
                    }
                    else    {
                        h1[i1][j1][bno]+=(vmag*20-vmag*bdiff)/20;
                        h1[i1][j1][(bno+1)%9]+=(vmag*bdiff)/20;
                    }
                }
            }
            j1++;
        }
        i1++;
    }
    
    //Normalizing histograms
    double *res=new double[756];
    normalize2(res,h1);*/
    printf("hello\n");
    return;
}
Mat skeleton(Mat img)   {
    Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    Mat temp(img.size(), CV_8UC1);
    bool done;
    do
    {
      morphologyEx(img, temp, cv::MORPH_OPEN, element);
      bitwise_not(temp, temp);
      bitwise_and(img, temp, temp);
      bitwise_or(skel, temp, skel);
      erode(img, img, element);
     
      double max;
      cv::minMaxLoc(img, 0, &max);
      done = (max == 0);
    } while (!done);
    
    return skel;
}
vector<float> featD(Mat m)  {
    SIFT sf;
	Mat desc;
	vector<KeyPoint> vc;
	sf(m,Mat(),vc,desc);
	vector<float> res;
    for(int i=0; i<desc.cols; i++)  {
        float v=0;
         for(int j=0; j<desc.rows; j++) {
            v+=desc.at<float>(j,i);
        }
        res.push_back(v);
    }
    return res;
}
void hog_train2(Mat m,Mat o,Rect r)
{
    vector<vector<float>> featV;    //store feature vectors of each character
    vector<vector<Point>> locs;     // store location of each character
    HOGDescriptor hog;
    hog.winSize=Size(rw,rh);
    int thresh=0;
    int *ch=new int[1000];
    int ind=0;
    int *a=new int[r.width];	
	for (int i = 0; i < r.width; i++) 
	{
		a[i]=0;
		for(int j=0;j<r.height;j++)	{
			Scalar myColor = m.at<uchar>(Point(i+r.x,j+r.y));
			int pixelValue = myColor.val[0]/255;
			a[i]+=pixelValue;
		}
	}
	int flag=1,strt=0,i;
	Mat drawing = Mat::zeros( m.size(), CV_8UC3 );
	o.copyTo(drawing);
	for(i=0;i<r.width;i++)
	{
		if(a[i]>thresh)
		flag=2;
		else if(a[i]<=thresh&&flag==2)
		{
			Rect rec(r.x+strt,r.y,i-strt,r.height);
			rectangle(drawing,rec, Scalar(0,0,255), 1);
			Mat tmp=m(rec);
			Mat cc= Mat::zeros( tmp.size(), CV_8UC1 );
			tmp.copyTo(cc); 
			
            Mat img=resize2(cc);
            vector<float> fv;
            vector<Point> lc;
            //fv=featD(img);
            hog.compute (img,fv,Size(8,8),Size(0,0),lc);
            featV.push_back(fv);
            locs.push_back(lc);
            char ctmp;
            int chh=0;
            int base=1;
            imshow("dd",img);
            while((ctmp=waitKey(0))!=' ' || chh==0)    {
                if(ctmp==8) {
                    if(base!=100) {
                        base=base/100;
                        chh=chh%base;
                    }
                    else    {
                        base =1;
                        chh=0;
                    }
                    continue;
                }
                if(ctmp<32)  continue;
                ctmp=ctmp%256 - 32;
                chh=(ctmp%256)*base+chh;
                base=base*100;
            }
            printf("%d\n",chh);
            ch[ind++]=chh;
			flag=1;	
			strt=i;
		}
		else if(a[i]<=thresh)
		{	flag=1; strt=i;	}
		namedWindow("char", CV_WINDOW_AUTOSIZE); 
		imshow("char",drawing);

	}
	if(strt!=i && flag==2)   {          //Checking end of word.
	
	    Rect rec(r.x+strt,r.y,i-strt,r.height);
		rectangle(drawing,rec, Scalar(0,0,255), 1);
		Mat tmp=m(rec);
		Mat cc= Mat::zeros( tmp.size(), CV_8UC1 );
		tmp.copyTo(cc); 
		
        Mat img=resize2(cc);
        vector<float> fv;
        vector<Point> lc;
        //fv=featD(img);
        hog.compute (img,fv,Size(8,8),Size(0,0),lc);
        featV.push_back(fv);
        locs.push_back(lc);
        char ctmp;
        int chh=0;
        int base=1;
        imshow("dd",img);
        while((ctmp=waitKey(0))!=' ' || chh==0)    {
            if(ctmp==8) {
                if(base!=100) {
                    base=base/100;
                    chh=chh%base;
                }
                else    {
                    base =1;
                    chh=0;
                }
                continue;
            }
            if(ctmp<32)  continue;
            ctmp=ctmp%256 - 32;
            chh=(ctmp%256)*base+chh;
            base=base*100;
        }
        printf("%d\n",chh);
        ch[ind++]=chh;
		flag=1;	
		strt=i;
	}
	imwrite("char.png",drawing);
    FileStorage fs2("train.yml", FileStorage::READ);
	Mat rfeaV, rLab;            //read from File Mats
	fs2["feav"] >> rfeaV;
	fs2["label"] >> rLab;
	fs2.release();
	int cols=featV[0].size();
	Mat feaV(featV.size(),cols,CV_32FC1);   //Mat Create during this call
    for(int i=0; i<feaV.rows; i++)
         for(int j=0; j<cols; j++)
              feaV.at<float>(i, j) = featV[i][j];
	Mat labMat(ind, 1, CV_32SC1, ch);
    printf("a: %d\n",feaV.rows);
    Mat mfeaV,mLab;                           //Merging read and created Mats
    if(!rfeaV.empty())
        vconcat(rfeaV,feaV,mfeaV);
    else    mfeaV=feaV;
    if(!rLab.empty())
        vconcat(rLab,labMat,mLab);
    else    mLab=labMat;
    FileStorage fs("train.yml", FileStorage::WRITE);
    fs << "feav" << mfeaV;
    fs << "label" << mLab;
    fs.release();
	
}

void hog_train(Mat m,Mat o,Rect r)
{
    vector<vector<float>> featV;    //store feature vectors of each character
    vector<vector<Point>> locs;     // store location of each character
    HOGDescriptor hog;
    hog.winSize=Size(rw,rh);
    int thresh=0;
    int ch[]={33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,17,18,19,20,21,22,23,24,25,16,27,7,12,14,13,8,9};
    int ind=69;
    int *a=new int[r.width];	
	for (int i = 0; i < r.width; i++) 
	{
		a[i]=0;
		for(int j=0;j<r.height;j++)	{
			Scalar myColor = m.at<uchar>(Point(i+r.x,j+r.y));
			int pixelValue = myColor.val[0]/255;
			a[i]+=pixelValue;
		}
	}
	int flag=1,strt=0,i;
	Mat drawing = Mat::zeros( m.size(), CV_8UC3 );
	o.copyTo(drawing);
	for(i=0;i<r.width;i++)
	{
		if(a[i]>thresh)
		flag=2;
		else if(a[i]<=thresh&&flag==2)
		{
			Rect rec(r.x+strt,r.y,i-strt,r.height);
			rectangle(drawing,rec, Scalar(0,0,255), 1);
			Mat tmp=m(rec);
			Mat cc= Mat::zeros( tmp.size(), CV_8UC1 );
			tmp.copyTo(cc); 
			
            Mat img=resize2(cc);
            vector<float> fv;
            vector<Point> lc;
            //fv=featD(img);
            hog.compute (img,fv,Size(8,8),Size(0,0),lc);
            featV.push_back(fv);
            locs.push_back(lc);
            //int chh=waitKey(0)%256;
            //printf("%d\n",chh);
            //ch[ind++]=chh;
			flag=1;	
			strt=i;
		}
		else if(a[i]<=thresh)
		{	flag=1; strt=i;	}
		namedWindow("char", CV_WINDOW_AUTOSIZE); 
		imshow("char",drawing);

	}
	if(strt!=i && flag==2)   {          //Checking end of word.
	
	    Rect rec(r.x+strt,r.y,i-strt,r.height);
		rectangle(drawing,rec, Scalar(0,0,255), 1);
		Mat tmp=m(rec);
		Mat cc= Mat::zeros( tmp.size(), CV_8UC1 );
		tmp.copyTo(cc); 
		
        Mat img=resize2(cc);
        vector<float> fv;
        vector<Point> lc;
        //fv=featD(img);
        hog.compute (img,fv,Size(8,8),Size(0,0),lc);
        featV.push_back(fv);
        locs.push_back(lc);
        //int chh=waitKey(0);
        //printf("%c\n",chh);
        //ch[ind++]=chh;
		flag=1;	
		strt=i;
	}
	imwrite("char.png",drawing);
	
    FileStorage fs2("train.yml", FileStorage::READ);
	Mat rfeaV, rLab;            //read from File Mats
	fs2["feav"] >> rfeaV;
	fs2["label"] >> rLab;
	fs2.release();
	int cols=featV[0].size();
	Mat feaV(featV.size(),cols,CV_32FC1);   //Mat Create during this call
    for(int i=0; i<feaV.rows; i++)
         for(int j=0; j<cols; j++)
              feaV.at<float>(i, j) = featV[i][j];
	Mat labMat(ind, 1, CV_32SC1, ch);
    printf("a: %d\n",feaV.rows);
    Mat mfeaV,mLab;                           //Merging read and created Mats
    if(!rfeaV.empty())
        vconcat(rfeaV,feaV,mfeaV);
    else    mfeaV=feaV;
    if(!rLab.empty())
        vconcat(rLab,labMat,mLab);
    else    mLab=labMat;
    FileStorage fs("train.yml", FileStorage::WRITE);
    fs << "feav" << mfeaV;
    fs << "label" << mLab;
    fs.release();
	
}

void hog(Mat m,Mat o,Rect r,char *c,int *ind)
{
    FileStorage fs("train.yml", FileStorage::READ);
    Mat rfeaV, rLab;            //read from File Mats
	fs["feav"] >> rfeaV;
	fs["label"] >> rLab;
	fs.release();
    
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    CvSVM SVM;
    SVM.train(rfeaV, rLab, Mat(), Mat(), params);
    
    HOGDescriptor hog;
    hog.winSize=Size(rw,rh);
    vector<vector<float>> featV;    //store feature vectors of each character
    vector<vector<Point>> locs;     // store location of each character
    int thresh=0;
    int *a=new int[r.width];	
	for (int i = 0; i < r.width; i++) 
	{
		a[i]=0;
		for(int j=0;j<r.height;j++)	{
			Scalar myColor = m.at<uchar>(Point(i+r.x,j+r.y));
			int pixelValue = myColor.val[0]/255;
			a[i]+=pixelValue;
		}
	}
	int flag=1,strt=0,i;
	Mat drawing = Mat::zeros( m.size(), CV_8UC3 );
	o.copyTo(drawing);
	for(i=0;i<r.width;i++)
	{
		if(a[i]>thresh)
		flag=2;
		else if(a[i]<=thresh&&flag==2)
		{
			Rect rec(r.x+strt,r.y,i-strt,r.height);
			rectangle(drawing,rec, Scalar(0,0,255), 1);
			Mat tmp=m(rec);
			Mat cc= Mat::zeros( tmp.size(), CV_8UC1 );
			tmp.copyTo(cc); 
			
            Mat img=resize2(cc);
            vector<float> fv;
            vector<Point> lc;
            hog.compute (img,fv,Size(8,8),Size(0,0),lc);
            
            //fv=featD(img);
            Mat fvm(fv,true);
            int val=SVM.predict(fvm);
            if(val<100)
                c[(*ind)++]=(char) (val+32);
            else    {
                int t22=val;
                while(t22>0)    {
                    int t33=(t22%100) + 32;
                    t22=t22/100;
                    c[(*ind)++]=(char) (t33%256);
                }
            }
			flag=1;	
			strt=i;
		}
		else if(a[i]<=thresh)
		{	flag=1; strt=i;	}
		namedWindow("char", CV_WINDOW_AUTOSIZE); 
		imshow("char",drawing);

	}
	if(strt!=i && flag==2)   {          //Checking end of word.
	
	    Rect rec(r.x+strt,r.y,i-strt,r.height);
		rectangle(drawing,rec, Scalar(0,0,255), 1);
		Mat tmp=m(rec);
		Mat cc= Mat::zeros( tmp.size(), CV_8UC1 );
		tmp.copyTo(cc); 
		
        Mat img=resize2(cc);
        vector<float> fv;
        vector<Point> lc;
        
        //fv=featD(img);
        hog.compute (img,fv,Size(8,8),Size(0,0),lc);
        Mat fvm(fv,true);
        int val=SVM.predict(fvm);
        if(val<100)
            c[(*ind)++]=(char) (val+32);
        else    {
            int t22=val;
            while(t22>0)    {
                int t33=(t22%100) + 32;
                t22=t22/100;
                c[(*ind)++]=(char) (t33%256);
            }
        }
		flag=1;	
		strt=i;
	}
	//imwrite("char.png",drawing);
	
}
