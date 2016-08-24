package edu.neu.coe.core;

import java.io.*;
import java.util.*;

/**
 * 
 * @author jiangyunwei
 *	
 */
public class SVMAlgorithmII {
	private static int numPeople=0;
	private static double[] transformedLabel;
	/*
	 *  first traverse each alpha in the data set and then select one
	 *  alpha in the remaining data set, build one alpha pair with
	 *  these 2 alphas. we need to update them simultaneously because
	 *  	of the constraint sum(α_i·label_i)=0. And thus we need to 
	 *  create one function to select one integer within a range
	 */
	private static int selectJRandomly(int i,int m)
	{
		Random rand = new Random();
		int j=i;
		while(j==i)
			j = rand.nextInt(m);
		return j;
	}
	private static double clipAlpha(double alpha_j,double H,double L)
	{
		if(alpha_j>H) alpha_j = H;
		else if(alpha_j<L) alpha_j = L;
		return alpha_j;
	}
	private static void transform(int[] trainY,int target)
	{
		transformedLabel = new double[trainY.length];
		for(int i=0;i<trainY.length;i++)
		{
			if(trainY[i]!=target)
			{
				transformedLabel[i] = -1;
			}else
			{
				transformedLabel[i] = 1;
			}
		}
	}
	/*
	 *  Pseudo-code:
	 *  1) Create one alpha vector and initialize it as a zero vector
	 *  2) while current iteration < max iteration:
	 *  			for each data vector of the data set:
	 *  				if the data vector can be optimized:
	 *  					select another data vector randomly
	 *  					optimize the two vectors simultaneously
	 *  					if the two vectors cannot be optimized:
	 *  						break;
	 * 			if no vectors get optimized, increase current iteration
	 */
	private static void SVM(double[][] trainX,int[] trainY,double C,double toler,int maxIteration,int target,int compared)
	{
		String path = System.getProperty("user.dir")+"/src/edu/neu/coe/resources/models/svm/weight"+target+"_"+compared+".txt";
		transform(trainY,target);// train a model for each class
		//System.out.println(Arrays.toString(transformedLabel));
		double b = 0;
		int m = trainX.length;
		int n = trainX[0].length;
		int iteration = 0;
		double[] alphas = new double[m];
		while(iteration<maxIteration)
		{
			int alphaPairsChanged = 0;// to record whether alpha optimized, if no pairs change, continue
			for(int i=0;i<m;i++)
			{
				
				//fXi = (alphas*Y)·K(x,x[i])+b = (alphas*Y)·(Φ(X)*Φ(X[i]))+b
				double fXi = Utils.vectorTimesVector(Utils.vectorMulVector(alphas, transformedLabel), Utils.matrixMulVector(Utils.linearKernel(trainX, 256),Utils.linearKernel(trainX[i], 256)))+b;
				// calculate errors
				double Ei = fXi - transformedLabel[i];
				// if the error Ei is large enough, then we should optimize the alpha vector,
				// we consider both the positive margin and negative margin in the if clause,
				// also we should make sure alpha meet the constraint 0<alpha<C because 
				// if alpha<0 will be adjusted to alpha=0 and alpha>C will be adjusted to
				// alpha = C, so if alpha is in {0,C}, it means that they are already on the
				// boundary, and cannot be increased or decreased, thus unnecessary to optimize
				if(transformedLabel[i]*Ei<-toler&&alphas[i]<C||transformedLabel[i]*Ei>toler&&alphas[i]>0)
				{
					int j = selectJRandomly(i,m);
					double fXj = Utils.vectorTimesVector(Utils.vectorMulVector(alphas, transformedLabel), Utils.matrixMulVector(Utils.linearKernel(trainX, 256),Utils.linearKernel(trainX[j], 256)))+b;
					double Ej = fXj - transformedLabel[j];
					//update alpha
					double alphaIOld = alphas[i];
					double alphaJOld = alphas[j];
					double L = 0, H = 0;
					/**
					 * 	According to the SMO algorithm
					 *  if Y[i] != Y[j]
					 *  		L = max(0,alphas[j]-alphas[i])
					 *  		H = min(C,C+alphas[j]-alphas[i])
					 *  else 
					 *  		L = max(0,alphas[j]+alphas[i]-C)
					 *  		H = min(C,alphas[j]+alphas[i])
					 */
					if(transformedLabel[i]!=transformedLabel[j])
					{
						L = Math.max(0, alphas[j]-alphas[i]);
						H = Math.min(C, C+alphas[j]-alphas[i]);
					}else
					{
						L = Math.max(0, alphas[j]+alphas[i]-C);
						H = Math.min(C, alphas[j]+alphas[i]);
					}
					// if L = H, go to next iteration;otherwise, update eta and alphas
					if(L==H) continue;
					//K(x1,x2) = Φ(x1)·Φ(x2), here Φ(x) = constant·x
					double eta = 2*Utils.vectorTimesVector(Utils.linearKernel(trainX[i], 256), Utils.linearKernel(trainX[j], 256))-Utils.vectorTimesVector(Utils.linearKernel(trainX[i], 256), Utils.linearKernel(trainX[i], 256)) - Utils.vectorTimesVector(Utils.linearKernel(trainX[j], 256), Utils.linearKernel(trainX[j], 256));
					/*
					 *  if eta=0, then we need to quit the for loop.
					 *  Here we simplify the SMO procedure because
					 *  if eta = 0, it would be quite complex to compute
					 *  the αj, so we just ignore the procedure.
					 *  Occasionally, we will have eta=0, so it is reasonable
					 *  to ignore these special situations
					 */
					if(eta==0.0) continue;
					/*
					 *  αiyi+αjyj=αi*yi+αj*yj=constant
					 *  do not consider constraint 0<=αj<=C
					 *  αj := αj*+yj(Ei-Ej)/(K(xi,xi)+K(xj,xj)-2K(xi,xj))
					 *  	   := αj+yj(Ei-Ej)/(Φ(Xi)*Φ(Xi)+Φ(Xj)*Φ(Xj)-2Φ(Xi)*Φ(Xj))
					 */
					alphas[j] -=transformedLabel[j]*(Ei-Ej)/eta;
					alphas[j] = clipAlpha(alphas[j],H,L);
					if(Math.abs(alphas[j]-alphaJOld)<0.0001) continue;// ignore small changes
					alphas[i] +=transformedLabel[j]*transformedLabel[i]*(alphaJOld-alphas[j]);
					/*
					 * α* means old α value
					 * b1 = b-Ei-yi(αi-αi*)K(xi,xi)-yj(αj-αj*)K(xi,xj)
					 * b2 = b-Ej-yi(αi-αi*)K(xi,xi)-yj(αj-αj*)K(xj,xj)
					 * K(xi,xj) = Φ(xi)·Φ(xj)
					 * 
					 * b :=
					 * 	if 0<αi<C then b1
					 * 	else if 0<αj<C then b2
					 * 	else (b1+b2)/2
					 */
					double b1 = b-Ei-transformedLabel[i]*(alphas[i]-alphaIOld)*(Utils.vectorTimesVector(Utils.linearKernel(trainX[i], 256),Utils.linearKernel(trainX[i], 256)))-transformedLabel[j]*(alphas[j]-alphaJOld)*Utils.vectorTimesVector(Utils.linearKernel(trainX[i], 256),Utils.linearKernel(trainX[j], 256));
					double b2 = b-Ej-transformedLabel[i]*(alphas[i]-alphaIOld)*(Utils.vectorTimesVector(Utils.linearKernel(trainX[i], 256),Utils.linearKernel(trainX[j], 256)))-transformedLabel[j]*(alphas[j]-alphaJOld)*Utils.vectorTimesVector(Utils.linearKernel(trainX[j], 256),Utils.linearKernel(trainX[j], 256));
					if(0<alphas[i]&&alphas[i]<C) b=b1;
					else if(0<alphas[j]&&alphas[j]<C) b=b2;
					else b=(b1+b2)/2.0;
					alphaPairsChanged+=1;
				}
			}
			//if there are changes of alpha, set current iteration 0, 
			// assuring that the code will only end in the case where 
			// no updates and it finishes the max iteration
			if(alphaPairsChanged==0) iteration+=1;
			else iteration = 0;
		}
		double[] weight = calculateWeights(alphas,transformedLabel,trainX);
		//Utils.save(weight, System.getProperty("user.dir")+"/src/edu/neu/coe/resources/model/svm/weight"+target+".txt");
		Utils.save(weight, b, path);
	}
	private static double[] calculateWeights(double[] alphas,double[] Y,double[][]X)
	{
		int m = X.length, n=X[0].length;
		double[] weights = new double[n];
		for(int i=0;i<m;i++)
		{
			// w := w + αYΦ(x)
			weights = Utils.vectorAddVector(weights, Utils.constantMulVector(alphas[i]*Y[i], Utils.linearKernel(X[i], 256)));
		}
		return weights;
	}
	public static void SVM(int num,double C,double toler,int maxIteration,int testNum)
	{
		File trainXFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/trainX.txt");
		File trainYFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/trainY.txt");
		
		double[][] trainX = Utils.ReadFile(trainXFile);
		int[] trainY = Utils.readFile(trainYFile);
		//System.out.println(Arrays.toString(trainY));
		// divide into groups
		List<double[][]> groups = new ArrayList<double[][]>();
		List<int[]> labelGroups = new ArrayList<int[]>();
		int index = 0;
		while(index<num)
		{
			List<double[]> group = new ArrayList<double[]>();
			List<Integer> labelGroup = new ArrayList<Integer>();
			for(int i=0;i<testNum;i++)
			{
				group.add(trainX[index*testNum+i]);
				labelGroup.add(trainY[index*testNum+i]);
			}
			//System.out.println(labelGroup);
			// we need to build a binary classification model for each 
			// 2 classes, thus it is necessary to have a combination
			double[][] groupArray = new double[group.size()][group.get(0).length];
			int[] labelGroupArray = new int[labelGroup.size()];
			for(int i=0;i<group.size();i++)
			{
				groupArray[i] = group.get(i);
				labelGroupArray[i] = labelGroup.get(i);
			}
			groups.add(groupArray);
			labelGroups.add(labelGroupArray);
			index++;
		}
		numPeople = num;
		for(int i=0;i<groups.size();i++)
		{
			for(int j=i+1;j<groups.size();j++)
			{
				//cnt++;
				double[][] groupI = groups.get(i);
				double[][] groupJ = groups.get(j);
				double[][] combinedGroup = Utils.combineTwo2DArray(groupI, groupJ);
				int[] labelGroupI = labelGroups.get(i);
				int[] labelGroupJ = labelGroups.get(j);
				int[] combinedLabels = Utils.combineTwoArray(labelGroupI, labelGroupJ);
				int target = labelGroupI[0];
				int compared =labelGroupJ[0];
				SVM(combinedGroup,combinedLabels,C,toler,maxIteration,target,compared);
			}
		}
	}
	private static int votePrediction(Map<Integer,Integer> cnt)
	{
		int vote = -1;
		int max = 0;
		Iterator<Integer> it = cnt.keySet().iterator();
		while(it.hasNext())
		{
			int key = it.next();
			int value = cnt.get(key);
			if(value>max)
			{
				vote = key;
				max = value;
			}
		}
		return vote;
	}
	public static List<ERROR> SVMTest()
	{
		List<ERROR> classificationErrors = new ArrayList<ERROR>();
		File testXFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/testX.txt");
		File testYFile = new File(System.getProperty("user.dir")+"/src/edu/neu/coe/resources/text/testY.txt");
		double[][] testX = Utils.ReadFile(testXFile);
		int[] testY = Utils.readFile(testYFile);
		System.out.println(Arrays.toString(testY));
		List<List<Integer>> positive = new ArrayList<List<Integer>>();
		List<Model> models = new ArrayList<Model>();
		String modelDirectory = System.getProperty("user.dir")+"/src/edu/neu/coe/resources/models/svm";
		File directoryFile = new File(modelDirectory);
		File[] files = directoryFile.listFiles();
		for(File file:files)
		{
			models.add(Utils.readModel(file));
		}
		int errorCnt = 0;
		for(int i=0;i<testX.length;i++)
		{
			double[] x = testX[i];
			Map<Integer,Integer> cnt = new TreeMap<Integer,Integer>();
			for(Model model:models)
			{
				double[] weight = model.weight;
				double sum = model.bias + Utils.vectorTimesVector(weight, Utils.linearKernel(x, 256));
				int predict = sum>0?model.preferedLabel:model.otherLabel;
				// because there are 780 models and thus it is necessary to
				// vote for the final prediction by applying majority voting
				if(!cnt.containsKey(predict))
				{
					cnt.put(predict, 1);
				}else
				{
					int newValue = cnt.get(predict)+1;
					cnt.replace(predict, newValue);
				}
			}
			int vote = votePrediction(cnt);
			int actual = testY[i];
			if(vote!=actual) 
			{
				errorCnt++;
				ERROR e = new ERROR(actual,vote);
				classificationErrors.add(e);
			}
			System.out.println("actual class is "+actual+", prediction is "+vote);
		}
		System.out.println("error rate = "+errorCnt*1.0/testX.length);
		return classificationErrors;
	}
}
