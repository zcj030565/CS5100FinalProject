package edu.neu.coe.opencv.UI;
import java.io.*;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

import edu.neu.coe.core.Utils;
import edu.neu.coe.opencv.Controller.*;
import javax.imageio.ImageIO;
import javax.swing.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import edu.neu.coe.opencv.utils.FileReader;
import edu.neu.coe.opencv.utils.ImageProcessor;
import libsvm.svm;
import libsvm.svm_model;
public class OpenCVGUI extends JFrame implements ActionListener
{
	static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
	private String illu = "";
	private JLabel imageView;
	private JTextArea illustration;
	private JScrollPane jsp;
	private JPanel imagePanel;
	private JLabel name;
	private JTextField nameText;
	private JPanel infoPanel;
	private JButton basicAlgorithm;
	private JButton register;
	private JButton recognize;
	private JButton training;
	private JPanel buttonPanel;
	private BufferedImage capturedImage;
	private VideoCapture capture;
	private CascadeClassifier faceDetector;
	private MatOfRect faceDetections;
	private svm_model loadedModel;
	private Map<Integer,String> map=new HashMap<Integer,String>();
	public void initGUI()
	{
		illu = Utils.readString(System.getProperty("user.dir")+"/src/ReadMe.txt");
		imageView = new JLabel();
		imagePanel = new JPanel();
		illustration = new JTextArea(18,20);
		illustration.setEditable(false);
		illustration.setText(illu);
		jsp = new JScrollPane(illustration);
		imagePanel.add(jsp, BorderLayout.WEST);
		imagePanel.add(imageView,BorderLayout.CENTER);
		imagePanel.setPreferredSize(new Dimension(400,300));
		name = new JLabel("Name",JLabel.CENTER);
		nameText = new JTextField();
		nameText.setPreferredSize(new Dimension(300,30));
		nameText.setEditable(false);
		infoPanel = new JPanel();
		infoPanel.add(name);
		infoPanel.add(nameText);
		basicAlgorithm = new JButton("Basic Algorithms Implemented By Ourselves");
		basicAlgorithm.addActionListener(this);
		register = new JButton("Register Face");
		register.addActionListener(this);
		recognize = new JButton("Recognize Face");
		recognize.addActionListener(this);
		training = new JButton("Training");
		training.addActionListener(this);
		buttonPanel = new JPanel();
		buttonPanel.add(basicAlgorithm);
		buttonPanel.add(register);
		buttonPanel.add(recognize);
		buttonPanel.add(training);
		this.add(imagePanel,BorderLayout.NORTH);
		this.add(infoPanel,BorderLayout.CENTER);
		this.add(buttonPanel,BorderLayout.SOUTH);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		//this.setDefaultCloseOperation(this.DISPOSE_ON_CLOSE);
		this.setVisible(true);
		this.setSize(800, 400);
		this.setResizable(false);
	}
	public void updateImage(Mat image)
	{
		BufferedImage img = ImageProcessor.toBufferedImage(image);
		capturedImage= img;
		imageView.setIcon(new ImageIcon(img));
	}
	public void captureVideo()
	{
		Mat webcamImage = new Mat();
		capture = new VideoCapture(0);
		capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 400);
		capture.set(Videoio.CAP_PROP_FRAME_HEIGHT,300);
		if(capture.isOpened())
		{
			while(true)
			{
				capture.read(webcamImage);
				if(!webcamImage.empty())
				{
					detectAndDrawFace(webcamImage);
					imageView.setIcon(new ImageIcon(ImageProcessor.toBufferedImage(webcamImage)));
				}else
				{
					 System.out.println("-- Frame not captured -- Break!");
					 break;
				}
			}
		}
	}
	public void loadCascade()
	{
		String path = System.getProperty("user.dir")+"/src/resources/frontalFace.xml";
		faceDetector = new CascadeClassifier(path);
	}
	public static void main(String[] args)
	{
		Mat image = Imgcodecs.imread("");
		OpenCVGUI gui = new OpenCVGUI();
		gui.initGUI();
		gui.loadCascade();
		gui.captureVideo();
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO Auto-generated method stub
		if(e.getSource().equals(register))
		{
			Mat curFrame = new Mat();
			capture.read(curFrame);
			Rect[] rects = faceDetections.toArray();
			
			if(rects.length==1&&rects[0].height>=140&&rects[0].width>=140)
			{
				curFrame=ImageProcessor.getRange(curFrame,rects[0]);
				RegisterGUI rgui = new RegisterGUI(curFrame);
			}else if(rects.length>1)
			{
				JOptionPane.showMessageDialog(null, "Too many faces detected" ,null, JOptionPane.INFORMATION_MESSAGE);
			}else if(rects.length==1)
			{
				JOptionPane.showMessageDialog(null, "The size shoud be no less than 140×140, your size is "+rects[0].height+"×"+rects[0].width+", please be closer to the screen" ,null, JOptionPane.INFORMATION_MESSAGE);
			}else
			{
				JOptionPane.showMessageDialog(null, "No face detected, please try again" ,null, JOptionPane.INFORMATION_MESSAGE);
			}
		}else if(e.getSource().equals(recognize))
		{
			if(loadedModel==null)
			{
				try {
					loadedModel = svm.svm_load_model(System.getProperty("user.dir")+"/src/model/finalModel");
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					FaceRecognition fr = new FaceRecognition(System.getProperty("user.dir")+"/src/images/");	
					loadedModel = fr.getFinalModel();
				}
			}
			Mat curFrame = new Mat();
			capture.read(curFrame);
			Rect[] rects = faceDetections.toArray();
			map=FileReader.readMapFile(System.getProperty("user.dir")+"/src/resources/texts/NameLabelMap.txt");
			nameText.setText("");
			if(rects.length==1&&rects[0].height>=140&&rects[0].width>=140)
			{
				curFrame=ImageProcessor.getRange(curFrame,rects[0]);
				
				curFrame = ImageProcessor.normalizeTarget(curFrame);
				FaceRecognition fr = new FaceRecognition();
				double[] label=fr.predict(curFrame,loadedModel,map);
				String predictedMan = map.get((int)label[0]);
				boolean stranger = Utils.isStranger(curFrame, predictedMan);
				if(stranger) predictedMan = "Stranger";
				nameText.setText(predictedMan);
			}
			else if(rects.length>1)
			{
				JOptionPane.showMessageDialog(null, "Too many faces detected" ,null, JOptionPane.INFORMATION_MESSAGE);
			}else if(rects.length==1)
			{
				JOptionPane.showMessageDialog(null, "The size shoud be no less than 140×140, your size is "+rects[0].height+"×"+rects[0].width+", please be closer to the screen" ,null, JOptionPane.INFORMATION_MESSAGE);
			}else
			{
				JOptionPane.showMessageDialog(null, "No face detected, please try again" ,null, JOptionPane.INFORMATION_MESSAGE);
			}
		}else if(e.getSource().equals(training))
		{
			File allImages = new File(System.getProperty("user.dir")+"/src/images/");
			if(allImages.list().length%10!=0)
			{
				JOptionPane.showMessageDialog(null, "Please register 10 images for each user" ,null, JOptionPane.INFORMATION_MESSAGE);
				return;
			}
			FaceRecognition fr = new FaceRecognition(System.getProperty("user.dir")+"/src/images/");	
			loadedModel = fr.getFinalModel();
		}else
		{
			MainInterface mi = new MainInterface();
		}
	}
	private void delay(int time)
	{
		long curMs = (new Date()).getTime();
		long nowMs = curMs;
		do
		{
			nowMs = (new Date()).getTime();
			
		}while((nowMs-curMs)<time);
	}
	private void detectAndDrawFace(Mat image) {
	    faceDetections = new MatOfRect();
	    faceDetector.detectMultiScale(image, faceDetections, 1.1, 4,0,new Size(20,20),new Size());//FACE参数
	    // Draw a bounding box around each face.
	    for (Rect rect : faceDetections.toArray()) 
	    {
	        Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));   
	    }
	    delay(200);
	}

}
