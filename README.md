# CS5100FinalProject

The project consists of 2 parts: The basic part covers the implementation of PCA, KNN, SVM, Logistic Regression algorithms. 
The advanced part is a face recognition system based on OpenCV.

Before Running face recognition system based on OpenCV, Please make sure you’ve configured the environment:
Install OpenCV 3.1.0.3 for java on your MacBook

1.  Open Terminal

2.	brew tap homebrew/science

3.	brew install opencv3 --with-java

4.	cd /usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/java

5.	cp libopencv_java310.so libopencv_java310.dylib

After installing OpenCV3, please configure the user library in Eclipse. 
The steps are following:

a.	Click on “Eclipse” icon on the up left corner and then select “Preferences”.

b.	At the “Preferences” menu, click on “Java” item, and then select “User Libraries” under “Build Path” submenu. 

c.	Click on “New” button and in text area and input your library name;

d.	Select your library name and click on “Add External JARS”, choose your OpenCV jar file. Then configure native path as /usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/java

Download Jama and libsvm and add them to project path.

https://drive.google.com/open?id=0B2KDlCdpH8iAb3ZzSWowSmp6NFE

For the implementation of KNN, SVM and logistic regression algorithms coded by ourselves:

a. Please click “Basic Algorithms Implemented By Ourselves” button.

b. Click “Training Image Process” button to process training images with PCA.

c. Click “Testing Image Process” button to process test images with PCA.

d. Click “KNN” button to run KNN algorithm.The outputs on the right side are errors.The outputs at Console window are complete output.

e. Click “SVM Model Training” button to train SVM models which are saved as files and can be reused.

f. Click “SVM” button to run SVM algorithm.

g. Click “Logistic Regression Model training” button.

h. Click “Logistic Regression” button.

For the face recognition system based on OpenCV: 

If you want to register new users,please click “Register Face”. It is required to save 10 images for each new user. Please remember to retrain your models by clicking “Training” button. 

If you want to recognize a user, please click “Recognize Face” button.
