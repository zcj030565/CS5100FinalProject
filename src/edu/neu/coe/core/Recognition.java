package edu.neu.coe.core;

import java.io.File;
import java.util.*;
import Jama.*;

public class Recognition {
	private static double power = 0.90;

	/**
	 * 
	 * @param samples
	 * @return the array of test faces
	 */
	public static double[][] getTestArray(List<double[]> samples) {
		double[][] result = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++)
			result[i] = samples.get(i);
		return result;
	}

	private static List<double[]> readTrainFaces(List<String> paths) {
		List<double[]> samples = new ArrayList<double[]>();
		for (String path : paths) {// transform file to array
			double[][] array = ImageProcessing.image2Array(path);
			double[] vector = Utils.matrix2Vector(array);//2D->1D
			samples.add(vector);// Each vector an image, 200 in total
		}
		return samples;
	}

	private static List<double[]> readTrainFaces(int numPeople, int start, int end) {
		List<String> paths = new ArrayList<String>();
		// append all paths of train images to the list
		for (int i = 0; i < numPeople; i++) {
			for (int j = start; j <= end; j++) {
				// build absolute path
				StringBuilder sb = new StringBuilder(
						System.getProperty("user.dir") + "/src/edu/neu/coe/resources/images/");
				String cls = String.valueOf(i) + "";
				sb.append(cls + "_" + j + ".bmp");
				// add absolute path to the list
				paths.add(sb.toString());
			}
		}
		List<double[]> trainFaces = readTrainFaces(paths);
		return trainFaces;
	}

	/**
	 * 
	 * @param trainFaces
	 * @param numPeople
	 * @param start
	 * @param end
	 * This method is used for extract features from images through PCA algorithm
	 */
	private static void PCA(List<double[]> trainFaces, int numPeople, int start, int end) {
		double[] mean = ImageProcessing.meanImage(trainFaces);// Get the average vector of train faces
// Subtracting the mean is equivalent to translating the coordinate system to the location of the mean
		Matrix xmean = ImageProcessing.subAverage(mean, trainFaces);
// Sigma is a square,symmetric covariance matrix
		Matrix sigma = xmean.times(xmean.transpose());
		EigenvalueDecomposition ed = sigma.eig();// EigenValue decomposition
		// Av = λv, A (non-zero) vector v of dimension N is an eigenvector of a square (N×N) matrix A
		// if it satisfies the linear equation where λ is a scalar, termed the eigenvalue corresponding to v.
		double[][] v = ed.getV().getArray();// Return the eigenvector matrix
		double[] d = ed.getRealEigenvalues();// Return the real parts of the eigenvalues ascend sorted
		// Each eigenvector respond to a eigenvalue.
		int[] index = new int[d.length];//
		for (int i = 0; i < index.length; i++)
			index[i] = i;
		// num of eigenvector
		int cols = v[0].length;
		// sort in descend order
		int M = xmean.getArray().length;
		int N = xmean.getArray()[0].length;
		double[][] vsort = new double[M][cols];
	    // vsort is of M*cols matrix,each column means an eigenvector
		double[] dsort = new double[M];// dsort saves eigenvalue in descend order
		// Eigenvector with largest eigenvalue captures the most variation among training vectors
		// These eigenvectors are known as principal component vectors
		for (int i = 0; i < cols; i++) {
			for (int j = 0; j < M; j++)
			vsort[j][i] = v[j][index[cols - i - 1]];
		// each column is an eigenvector
			dsort[i] = d[index[cols - i - 1]];
		}
		// Reserve no less than 90% of image information
		double dsum = Utils.sum(dsort);
		double dsum_extract = 0;
		int p = -1;
		while (dsum_extract / dsum < power) {
			p += 1;
			dsum_extract = Utils.sum(dsort, 0, p);
		}
		int i = 0;
		double[][] base = new double[N][p + 1];// base is a matrix of N*(p+1),element = dsort[i]^(-1/2)
		while (i <= p && dsort[i] > 0) {
			double[][] vsort_i = new double[M][1];
			for (int k = 0; k < M; k++) {
				vsort_i[k][0] = vsort[k][i];// Get first p columns only, one column one time.
			}
			double[][] tmp = xmean.transpose().times(new Matrix(vsort_i)).times(Math.pow(dsort[i], -0.5)).getArray();
			// tmp = (10304*200) * (200*1) * [eigenvalues^(-1/2)]
			for (int k = 0; k < N; k++) {
				base[k][i] = tmp[k][0];
			}
			i += 1;
		}
		// Size of base is 10304*75
		// project training samples to the computed coordinate and get a matrix of (200*75)
		double[][] projector = ImageProcessing.projectImageToCoordinate(trainFaces, base);
		// normalize all data to [-1,1]
		double[][] S = Utils.unitNormalize(projector);// 200*75
		int[] labels = new int[trainFaces.size()];
		int labelIndex = 0;
		int count = 0;
		// save labels of each image
		while (labelIndex < numPeople) {
			for (int kk = start; kk <= end; kk++) {
				labels[count++] = labelIndex;
			}
			labelIndex++;
		}
		// save file
		Utils.save(S, System.getProperty("user.dir") + "/src/edu/neu/coe/resources/text/trainX.txt");
		Utils.save(labels, System.getProperty("user.dir") + "/src/edu/neu/coe/resources/text/trainY.txt");
		Utils.save(base, System.getProperty("user.dir") + "/src/edu/neu/coe/resources/text/base.txt");
	}

	public static void train(int numPeople, int start, int end)

	{
		List<double[]> trainFaces = readTrainFaces(numPeople, start, end);
		PCA(trainFaces, numPeople, start, end);// dimensionality reduction
	}

	public static void test(int numPeople, int start, int end) {
		File baseFile = new File(System.getProperty("user.dir") + "/src/edu/neu/coe/resources/text/base.txt");
		if (!baseFile.exists()) {
			System.out.println("File not exists, please re-train the algorithm");
			return;
		}
		List<double[]> testFaces = new ArrayList<double[]>();
		double[][] base = Utils.ReadFile(baseFile);
		List<String> paths = new ArrayList<String>();
		List<Integer> allCls = new ArrayList<Integer>();
		for (int i = 0; i < numPeople; i++) {
			for (int j = start; j <= end; j++) {
				// build absolute path
				StringBuilder sb = new StringBuilder(
						System.getProperty("user.dir") + "/src/edu/neu/coe/resources/images/");
				String cls = String.valueOf(i) + "";
				allCls.add(i);
				sb.append(cls + "_" + j + ".bmp");
				// add absolute path to the list
				paths.add(sb.toString());
			}
		}
		int[] labels = new int[allCls.size()];
		for (int i = 0; i < allCls.size(); i++)
			labels[i] = allCls.get(i);
		for (String path : paths) {
			double[][] array = ImageProcessing.image2Array(path);
			double[] vector = Utils.matrix2Vector(array);
			double[][] vector2D = new double[1][];
			vector2D[0] = vector;
			double[] pca = (new Matrix(vector2D)).times(new Matrix(base)).getArray()[0];
			testFaces.add(pca);
			double[][] allCoor = getTestArray(testFaces);
			double[][] T = Utils.unitNormalize(allCoor);
			Utils.save(T, System.getProperty("user.dir") + "/src/edu/neu/coe/resources/text/testX.txt");
		}
		Utils.save(labels, System.getProperty("user.dir") + "/src/edu/neu/coe/resources/text/testY.txt");
	}
}
