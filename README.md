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

a.	Click on the “Eclipse” icon on the upper left corner and select “Preferences” in the list;

b.	In the “Preferences” menu, click on the “Java” item, and then select “User Libraries” under the “Build Path” submenu. 

c.	Click on “New” button and in the text area, input your library name;

d.	Select your library name and click on “Add External JARS”, choose the your OpenCV jar file. 
Later, configure the native path as /usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/java

Download Jama and libsvm and add them to the project path.

https://drive.google.com/open?id=0B2KDlCdpH8iAb3ZzSWowSmp6NFE
