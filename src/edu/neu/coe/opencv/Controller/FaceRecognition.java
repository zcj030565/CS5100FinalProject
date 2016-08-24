package edu.neu.coe.opencv.Controller;
import java.util.*;

import javax.imageio.ImageIO;
import javax.swing.JOptionPane;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.*;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import edu.neu.coe.opencv.utils.*;
import edu.neu.coe.opencv.utils.FileReader;
public class FaceRecognition {

	private String[] files;
	private String path;
	private svm_model finalModel=new svm_model();
	private DataProcessing dp = new DataProcessing();
	
	public svm_model getFinalModel() 
	{
		return finalModel;
	}
	public FaceRecognition()
	{
		
	}
	public FaceRecognition(String path)
	{
		File directory = new File(path);
		files = directory.list();
		this.path = path;
		finalModel=trainModel();
		try {
			svm.svm_save_model(System.getProperty("user.dir")+"/src/model/finalModel", finalModel);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			JOptionPane.showMessageDialog(null, "Please register 10 images for each user" ,null, JOptionPane.INFORMATION_MESSAGE);
		}
	}
	public svm_model trainModel(double[][] xtrain, double[][] ytrain)
	{
		svm_problem problem = new svm_problem();
		int recordCount = xtrain.length;
	
		int featureCount = xtrain[0].length;
		problem.y = new double[recordCount];
		problem.l = recordCount;
		problem.x = new svm_node[recordCount][featureCount];
		for(int i=0;i<recordCount;i++)
		{
			double[] features = xtrain[i];
			problem.x[i] = new svm_node[featureCount];
			for(int j=0;j<featureCount;j++)
			{
				svm_node node = new svm_node();
				node.index = j;
				node.value = features[j];
				problem.x[i][j] = node;
			}
			problem.y[i] = ytrain[i][0];
		}
	
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.C = 5;
		param.gamma = 10000;
		svm_model model = svm.svm_train(problem, param);
		
		return model;
	}
	public double[] predict(Mat curFrame,svm_model model,Map<Integer,String> map)
	{
		Imgproc.cvtColor(curFrame, curFrame, Imgproc.COLOR_BGR2GRAY);
		BufferedImage image=ImageProcessor.toBufferedImage(curFrame);
		double[][] pcaImg = dp.processing(image);
		double[][] normalized = dp.unitNormalize(pcaImg);
		
		int recordCount = normalized.length;
		int featureCount = normalized[0].length;
		double[] predictedY = new double[recordCount];
		for(int k=0;k<recordCount;k++)
		{
			double[] fVector = normalized[k];
			svm_node[] nodes = new svm_node[featureCount];
			for(int i=0;i<featureCount;i++)
			{
				svm_node node = new svm_node();
				node.index = i;
				node.value = fVector[i];
				nodes[i] = node;
			}
			int totalClasses = map.size();       
	        int[] labels = new int[totalClasses];
	        svm.svm_get_labels(model,labels);
	       
	        double[] prob_estimates = new double[totalClasses];
	        predictedY[k] = svm.svm_predict_probability(model, nodes, prob_estimates);
	      
		}
        return predictedY;
	}
	public svm_model trainModel()
	{
		DataProcessing dp = new DataProcessing();
		dp.processing(files,path);
		double[][] xtrain = FileReader.getFile(System.getProperty("user.dir")+"/src/resources/texts/trainX.txt");
		double[][] ytrain = FileReader.getFile(System.getProperty("user.dir")+"/src/resources/texts/trainY.txt");
		svm_model model = trainModel(xtrain, ytrain);
		return model;
	}
}
