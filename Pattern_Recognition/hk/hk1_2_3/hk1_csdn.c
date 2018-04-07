int main()  
{  
    //正常图像,构造函数不指定参数时，默认识别第一张图
    //构造函数可以指定识别第几张图，下面以第三张为例
    Picture pic = Picture(3);
    pic.startRecognize();

    //识别有噪声图像
    noisyPic noisyPic;
    noisyPic.startRecognize();

    //识别有划痕图像
    dirtyPic dirtyPic;
    dirtyPic.startRecognize();

    //识别放大缩小图像
    scalePic scale = scalePic(1);
    scale.startRecognize();

    return 0;  
}  


void preProcess(){          //自适应二值化&中值滤波
    Mat out;
    //自适应二值化
    adaptiveThreshold(source, source, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, adaptiveBiSize, adaptiveBiParam); 
    //中值滤波
    namedWindow("binary");
    imshow("binary",source);
    waitKey(0);
    medianBlur( source, out, medianBlurSize);
    namedWindow("medianblur");
    imshow("medianblur",out);
    waitKey(0);
    source = out;
    srcResult = out;  //用来显示
}


bool match(Mat src){
    int srcW,srcH,templatW, templatH, curtemplatW,curtemplatH,resultH, resultW; 
    Mat templat,result;
    srcW = src.cols;  
    srcH = src.rows;
    double currentMin = 1;
    int    currentIndex=0;
    double minValue, maxValue;  
    Point minLoc, maxLoc,matchLoc; 
    /*
    ** 相似度计算方法
    ** 0：CV_TM_SQDIFF        平方差匹配法，最好的匹配值为0；匹配越差，匹配值越大
    ** 1：CV_TM_SQDIFF_NORMED 归一化平方差匹配法
    ** 2：CV_TM_CCORR         相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好
    ** 3：CV_TM_CCORR_NORMED  归一化相关匹配法
    ** 4：CV_TM_CCOEFF        相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
    ** 5：CV_TM_CCOEFF_NORMED 归一化相关系数匹配法
    */
    int methodType=1;
    //循环判断8个数字哪个数字模板最为接近被测试图像
    for (int i=0;i<8;i++){
        templat = templatVec[i];
        templatW = templat.cols;  
        templatH = templat.rows;  
        if(srcW < templatW || srcH < templatH)  
        {  
            cout <<"模板不能比原图像大" << endl;  
            return 0;  
        }  
        resultW = srcW - templatW + 1;  
        resultH = srcH - templatH + 1;  
        result = cvCreateImage(cvSize(resultW, resultH), 1, 1);  

        matchTemplate(src, templat, result, methodType);   

        minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc,Mat() );
        //如果比当前最小还小，则储存该值,下标和坐标
        if (minValue<currentMin){
            currentMin = minValue;
            currentIndex=i;
            matchLoc.x=minLoc.x+window_x;
            matchLoc.y=minLoc.y+window_y;
            curtemplatW = templatW;
            curtemplatH = templatH;
        }
    }
    //cout<<"Min:"<<currentMin<<endl;
    //最小值比设定阈值小，则判断识别出这个数字
    if (currentMin<threshold){
        numresult.push_back(index[currentIndex]);
        cout<<"第"<<countnumbers<<"个数字是："<<index[currentIndex]<<endl;
        /*cout<<"左上角坐标为：("<<matchLoc.x<<","<<matchLoc.y<<")"<<endl;
        cout<<"右上角坐标：（"<<matchLoc.x+templatW<<","<<matchLoc.y<<")"<<endl;
        cout<<"左下角坐标：（"<<matchLoc.x<<","<<matchLoc.y+templatH<<")"<<endl;*/
        countnumbers++;
        rectangle(srcResult, matchLoc, cvPoint(matchLoc.x + curtemplatW, matchLoc.y+ curtemplatH), cvScalar(0,0,255));
        /*namedWindow("tmpresult");
        imshow("tmpresult",srcResult);
        waitKey(0);*/
        window_x =matchLoc.x+curtemplatW-1;
        return true;
    }
    //比阈值大则判定为非字符，扫描窗右移一个单位
    window_x++;
    return false;

}

virtual void processScan(){
    sourceW = source.cols;
    sourceH = source.rows;
    window_x = 0;
    window_y = 3;
    //加十以提高容错率
    bool last = false;
    while(window_x<sourceW-scanWindowW+5){
        if (window_x+scanWindowW>sourceW){
            window_x = sourceW - scanWindowW;
            last = true;
        }
        Mat tmp = scanWindow(window_x,window_y);
        match(tmp);
        if (last) break;
    }
    window_x = 30;
    scanWindowH = 35;
    window_y=sourceH - scanWindowH;
    while (window_x<=sourceW - scanWindowW-10){

        Mat tmp = scanWindow(window_x,window_y);
        match(tmp);
    }

}

//识别有噪点的图像
class noisyPic:public Picture{
    public:
    noisyPic(){
        Picture();
        threshold = 0.5;
        path="test\\noisy.bmp";
        adaptiveBiSize = 17;
        adaptiveBiParam= 19;
        medianBlurSize = 5;
        scanWindowW = 38;
        scanWindowH = 38;
    }
    void displayResult(){
        cout<<"当前识别的是有噪点的图像，识别结果为："<<endl;
        for (unsigned int i=0;i<numresult.size();i++){cout<<numresult[i]<<" ";}
        cout<<endl;
        cout<<"====================================================="<<endl;
        namedWindow("final");  
        imshow("final", srcResult);  
        waitKey(0); 
    }

};

//有划痕的图像
class dirtyPic:public Picture{
    public:
    dirtyPic(){
        Picture();
        threshold = 0.48;
        path="test\\dirty.bmp";
        adaptiveBiSize = 21;
        adaptiveBiParam= 23;
        medianBlurSize = 7;
        scanWindowW = 36;
        scanWindowH = 38;
    }
    virtual void displayResult(){
        cout<<"当前识别的是有划痕的图像，识别结果为："<<endl;
        for (unsigned int i=0;i<numresult.size();i++){cout<<numresult[i]<<" ";}
        cout<<endl;
        cout<<"====================================================="<<endl;
        namedWindow("final");  
        imshow("final", srcResult);  
        waitKey(0); 
    }
};

