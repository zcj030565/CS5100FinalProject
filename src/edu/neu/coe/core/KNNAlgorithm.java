package edu.neu.coe.core;
import java.util.*;
import java.io.*;
public class KNNAlgorithm 
{
	private static int K = 1;
	private static int numPeople=0;
	private static double calculateEuclideanDistance(double[] vecA,double[] vecB)
	{
		double distance = 0;
		for(int i=0;i<vecA.length;i++)
		{
			distance += Math.sqrt(Math.pow(vecA[i]-vecB[i], 2));
		}
		return distance;
	}
	private static int votePrediction(Map<Double,List<Integer>> classification)
	{
		int count = K;
		boolean finished = false;
		Iterator<Double> it = classification.keySet().iterator();
		List<Integer> allCls = new ArrayList<Integer>();
		while(!finished&&it.hasNext())
		{
			double key = it.next();//Key is the distance 
			List<Integer> cls = classification.get(key);//All of lables are attached to its distance
			// if the number of candidates is more than K, then put first K lable into allCLs
			if(cls.size()>=count)// Find nearest K image already
			{
				allCls.addAll(cls.subList(0, count));
				finished = true;
			}else//Less than K
			{
				allCls.addAll(cls);
				count -=cls.size();// Continue Looking for the rest
			}
		}
		int[] times = new int[numPeople+1];
		int max = 0;
		int vote = -1;
		for(int cls:allCls)
		{
			times[cls]++;
		}
		for(int i=0;i<times.length;i++)//Find which lable has majority
		{
			if(times[i]>max)
			{
				max = times[i];
				vote = i;
			}
		}
		return vote;
	}
	private static List<ERROR> KNN(double[][] trainX,int[] trainY,double[][] testX,int[] testY)
	{
		List<ERROR> classificationErrors = new ArrayList<ERROR>();
		
		long startTime = (new Date()).getTime();
		int errorCount = 0;
		for(int i=0;i<testX.length;i++)
		{
			double[] testx = testX[i];
			Map<Double,List<Integer>> classification = new TreeMap<Double,List<Integer>>();
			for(int j=0;j<trainX.length;j++)
			{
				double[] trainx = trainX[j];
				double distance = calculateEuclideanDistance(trainx,testx);
				if(!classification.containsKey(distance))//New distance

				{
					List<Integer> labels = new ArrayList<Integer>();
					labels.add(trainY[j]);
					classification.put(distance, labels);//Put All of images' lables with same distance together
				}else
				{
					classification.get(distance).add(trainY[j]);//Add to existing distance
				}
			}
			int vote = votePrediction(classification);
			int actual = testY[i];
			System.out.println("actual class is "+actual+", prediction is "+vote);
			if(vote!=actual)
			{
				ERROR e = new ERROR(actual,vote);
				classificationErrors.add(e);
				errorCount++;
			}
		}
		long endTime = (new Date()).getTime();
		System.out.println("Total classification time: "+(endTime-startTime)+"ms");
		System.out.println("error rate is "+errorCount*1.0/testX.length);
		return classificationErrors;
	}
	public static List<ERROR> KNN(int num)
	{
		File trainXFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/trainX.txt");
		File trainYFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/trainY.txt");
		File testXFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/testX.txt");
		File testYFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/testY.txt");
		double[][] trainX = Utils.ReadFile(trainXFile);
		int[] trainY = Utils.readFile(trainYFile);
		double[][] testX = Utils.ReadFile(testXFile);
		int[] testY = Utils.readFile(testYFile);
		numPeople = num;
		return KNN(trainX,trainY,testX,testY);
	}
}
