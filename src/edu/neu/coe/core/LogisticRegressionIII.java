package edu.neu.coe.core;

import java.io.*;
import java.util.*;
/**
 * 
 * @author jiangyunwei
 *
 */
class LRModel
{
	int label;
	double[] weight;
	public LRModel(int label, double[] weight)
	{
		this.label = label;
		this.weight = weight;
	}
}
public class LogisticRegressionIII 
{
	
	private static double[][] addOneColumn(double[][] trainX)
	{
		int numRecords = trainX.length;
		int numFeatures = trainX[0].length;
		double[][] result = new double[numRecords][numFeatures+1];
		for(int i=0;i<numRecords;i++)
		{
			double[] res = result[i];
			res[0] = 1;
			for(int j=1;j<=numFeatures;j++)
			{
				res[j] = trainX[i][j-1];
			}
		}
		return result;
	}
	private static double[] transform(int[] trainY,int target)
	{
		double[] transformed = new double[trainY.length];
		for(int i=0;i<trainY.length;i++)
		{
			if(trainY[i]==target)
				transformed[i] = 1;
			else
				transformed[i] = 0;
		}
		return transformed;
	}
	private static double sigmoid(double val)
	{
		return 1.0/(1+Math.exp(-val));
	}
	private static double[] sigmoid(double[] vector)
	{
		double[] result = new double[vector.length];
		for(int i=0;i<result.length;i++)
			result[i] = sigmoid(vector[i]);
		return result;
	}
	private static void LogisticRegression(double[][] trainX,int[] trainY,int maxIteration,int target)
	{
		double[] transformed = transform(trainY,target);
		//System.out.println(Arrays.toString(transformed));
		int numRecords = trainX.length;
		int numFeatures = trainX[0].length;
		/*
		 * 	z = w0 + w1x1 + w2x2 +... + wnxn
		 * 	  = w0*1 + w1x1 + w2x2 +... + wnxn
		 *  thus add one column at the begining
		 */
		double[][] newTrainX = addOneColumn(Utils.linearKernel(trainX, 256));
		//double[][] newTrainX = Utils.linearKernel(addOneColumn(trainX),256);
		//System.out.println(Arrays.toString(newTrainX[0]));
		double[] weight = new double[numFeatures+1];
		Arrays.fill(weight, 1.0);
		double learningRate = 0.01;
		/*	batch gradient descend algorithm
		 * 	J(w) = 0.5*sum(sigmoid(w·(β*x))-y)^2
		 *	w := w - α*J(w)' ==> w := w- α(sigmoid(w*(β*x))-y)*(β*x)
		 *	==> w := w - α*error*(β*x)
		 *
		 */	
		
		for(int iteration=0;iteration<maxIteration;iteration++)
		{
			double[] h = sigmoid(Utils.matrixMulVector(newTrainX, weight));
			//System.out.println(Arrays.toString(h));
			double[] error = Utils.vectorSubVector(transformed,h);
			weight = Utils.vectorAddVector(weight, Utils.constantMulVector(learningRate, Utils.matrixMulVector(Utils.transpose(newTrainX), error)));

		}
		String path = System.getProperty("user.dir")+"/src/edu/neu/coe/resources/models/lr2/"+target+".txt";
		Utils.save(weight, path);
	}
	public static void LogisticRegression(int num,int maxIteration,int testNum)
	{
		File trainXFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/trainX.txt");
		File trainYFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/trainY.txt");
		double[][] trainX = Utils.ReadFile(trainXFile);
		int[] trainY = Utils.readFile(trainYFile);
		// build one model for each class
		for(int target=0;target<num;target++)
		{
			LogisticRegression(trainX,trainY,maxIteration,target);
		}
	}
	public static List<ERROR> LogisticRegressionTest()
	{
		List<ERROR> classificationErrors = new ArrayList<ERROR>();
		File testXFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/testX.txt");
		File testYFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/testY.txt");
		double[][] testX = Utils.ReadFile(testXFile);
		//double[][] newTestX =  Utils.linearKernel(addOneColumn(testX),256);
		double[][] newTestX = addOneColumn(Utils.linearKernel(testX,256));
		int[] testY = Utils.readFile(testYFile);
		List<LRModel> models = Utils.readLRModels(new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/models/lr2"));
		int errorCnt = 0;
		for(int i=0;i<newTestX.length;i++)
		{
			double[] test = newTestX[i];
			int result = -1;
			double max = 0;
			/*
			 * 	for each test instance, try to compute the possibility
			 *  with each model, then select the model with greatest
			 *  possibility as the final result
			 */
			for(LRModel model:models)
			{
				double[] weight = model.weight;
				double possibility = sigmoid(Utils.vectorTimesVector(weight, test));
				if(possibility>max)
				{
					max = possibility;
					result = model.label;
				}
			}
			if(result!=testY[i]) 
			{
				errorCnt++;
				ERROR e = new ERROR(testY[i],result);
				classificationErrors.add(e);
			}
			System.out.println("real class is "+testY[i]+"---"+", prediction is "+result);
		}
		System.out.println("error rate is "+ errorCnt*1.0/testX.length);
		return classificationErrors;
	}
}
