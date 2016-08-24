Important:
Before running the program, please 
make sure that you have already 
installed OpenCV3 for Java. If 
not, please use the command 
"brew install opencv3 --with java", 
but please make sure you have 
already install homebrew on your 
macbook. After installing OpenCV3,
please add user library in your
Eclipse. Steps are:
Eclipse --> Preference -->
Java --> Build path --> User
libraries --> New --> Add 
External Jars --> native library
location.

There are 4 buttons in the GUI, 
if you click on the button 
"Basic Algorithms Implemented By 
Ourselves", you will see the KNN, 
SVM, Logistic Regression implemented 
by ourselves without using any 
third-party machine learning libraries.
These algorithms are running on the 
dataset ORL.

The other 3 buttons are for the face 
recognition system running on OpenCV 
and libsvm(a third-party machine 
learning library for SVM). You can 
register images for new person to 
enlarge your database if you click on 
the register button. You can ask the 
system to recognize the person who has 
already registered in the database. 
For those who has never been registered,
they may classified as strangers. Every 
time you register any new faces, 
please click on the training button to 
retrain your model, otherwise, it will
not make sense.