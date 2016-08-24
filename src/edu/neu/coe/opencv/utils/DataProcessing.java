package edu.neu.coe.opencv.utils;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;
import javax.swing.JOptionPane;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class DataProcessing {

	List<double[]> allTrainingSamples = new ArrayList<double[]>();
	Map<String,Integer> map = new HashMap<String,Integer>();
	List<String> records = new ArrayList<String>();
	public void print(double[] b,int start,int end)
	{
		for(int i=start;i<=end;i++)
			System.out.print(b[i]+" ");
		System.out.println();
	}
	//
	public double vectorMean(double[] vec)
	{
		double sum = 0;
		for(double x:vec)
			sum+=x;
		return sum/vec.length;
	}
	public double[] normalizeVector(double[] vec)
	{
		double[] result = vec.clone();
		double mean = vectorMean(vec);
		double var = vectorVar(vec);
		for(int i=0;i<vec.length;i++)
			result[i] = (vec[i]-mean)/var;
		return result;
	}
	public double vectorVar(double[] vec)
	{
		double mean = vectorMean(vec);
		double sum=0;
		for(double x:vec)
		{
			sum+=Math.pow(x-mean, 2);
		}
		return sum/(vec.length-1);
	}
	//
	public double[] vectorSubVector(double[] vecA,double[]vecB)
	{
		double[] result = new double[vecA.length];
		for(int i=0;i<vecA.length;i++)
			result[i]=vecA[i]-vecB[i];
		return result;
	}
	//
	public double sum(double[] vec)
	{
		double s=0;
		for(double elem:vec)
			s+=elem;
		return s;
	}
	public double sum(double[] vec,int start,int end)
	{
		double s=0;
		for(int i=start;i<=end;i++)
			s+=vec[i];
		return s;
	}
	//
	public double[][] getAllCoor(double[][] base)
	{
		double[][] matrix = new double[allTrainingSamples.size()][];
		for(int i=0;i<allTrainingSamples.size();i++)
			matrix[i] = allTrainingSamples.get(i);
		Matrix res = (new Matrix(matrix)).times(new Matrix(base));
		return res.getArray();
	}
	//Read the grayscale of images
	public double[][] image2Array(String path)
	{
		File file = new File(path);
		BufferedImage bi = null;
		int[] rgb = new int[3];
		if(file.exists()&&file.isFile())
		{
			try
			{
				bi = ImageIO.read(file);
				int width = bi.getWidth();
				int height = bi.getHeight();
				double[][] map = new double[height][width];
				int minx = bi.getMinX();
				int miny = bi.getMinY();
				for(int x = minx;x<minx+width;x++)
				{
					for(int y=miny;y<miny+height;y++)
					{
						int pixel = bi.getRGB(x, y);
						rgb[2] = (pixel&0xff);
						map[y][x]=rgb[2];
					}
				}
				return map;
			}catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		return null;
	}
	public double[][] image2Array(BufferedImage image)
	{
		int[] rgb = new int[3];
			try
			{
				int width = image.getWidth();
				int height = image.getHeight();
				double[][] map = new double[height][width];
				int minx = image.getMinX();
				int miny = image.getMinY();
				for(int x = minx;x<minx+width;x++)
				{
					for(int y=miny;y<miny+height;y++)
					{
						int pixel = image.getRGB(x, y);
						rgb[2] = (pixel&0xff);
						map[y][x]=rgb[2];
					}
				}
				return map;
			}catch(Exception e)
			{
				e.printStackTrace();
			}
		return null;
	}
	//
	public double[] matrix2Vector(double[][] matrix)
	{
		if(matrix==null) return null;
		int height = matrix.length;
		int width = matrix[0].length;
		int vecSize = height*width;
		double[] vector = new double[vecSize];
		int index = 0;
		//Column first, the row, left->right
		for(int i=0;i<width;i++)
		{
			for(int j=0;j<height;j++)
			{
				vector[index] = matrix[j][i];
				index++;
			}
		}
		return vector;
	}
	
	//average images
	public double[] mean()
	{
		int size = allTrainingSamples.get(0).length;
		double[] result = new double[size];
		int numRecords = allTrainingSamples.size();
		for(int i=0;i<size;i++)
		{
			double sum=0;
			for(int j=0;j<numRecords;j++)
				sum+=allTrainingSamples.get(j)[i];
			result[i]=sum/numRecords;
		}
		return result;
	}
	
	//subtract the average from each image
	public Matrix subAverage(double[] mean)
	{
		List<double[]> list = new ArrayList<double[]>();
		for(int i=0;i<allTrainingSamples.size();i++)
		{
			double[] sample = allTrainingSamples.get(i);
			list.add(vectorSubVector(sample,mean));
		}
		double[][] result = new double[list.size()][list.get(0).length];
		for(int i=0;i<list.size();i++)
			result[i] = list.get(i);
		return new Matrix(result);
	}
	//M*M'
	public void readTrainFaces(String[] files,String  path)
	{
		int index = 1;
		for(String file:files)
		{
			String absolutePath = path+file;
			double[][] a = image2Array(absolutePath);
			
			double[] b = matrix2Vector(a);
			allTrainingSamples.add(b);
			String name = file.split("_")[0];
			records.add(name);
			if(!map.containsKey(name))
			{
				map.put(name, index);
				index++;
			}
		}
	}

	public double[][] unitNormalize(double[][] allCoor)
	{
		double[][] x = allCoor.clone();
		int m = x.length;
		int n = x[0].length;
		double[][] S = new double[m][n];
		for(int i=0;i<m;i++)
		{
			S[i]=normalizeVector(x[i]);
		}
		return S;
	}
	
	public double[][] processing(BufferedImage image)
	{
		String path = System.getProperty("user.dir")+"/src/resources/texts/Base.txt";
		double[][] base = FileReader.getFile(path);
		
		double[][] a = image2Array(image);
		double[] b = matrix2Vector(a);
		//c= b * base;计算坐标，是1×p阶矩阵
		double[][]b2D = new double[1][];
		
		b2D[0]=b;
		double[] c = (new Matrix(b2D)).times(new Matrix(base)).getArray()[0];
		double[][] result = new double[1][c.length];
		result[0] = c;
		return result;
		
		//System.out.println(base);
	}
	public void processing(String[] files,String path)
	{
		readTrainFaces(files,path);
		double[] mean = mean();
		Matrix xmean = subAverage(mean);
		Matrix sigma = xmean.times(xmean.transpose());
		EigenvalueDecomposition ed = sigma.eig();
		double[][] v = ed.getV().getArray();
		double[] d1 = ed.getRealEigenvalues();
		//升序排序
		double[] d2 = d1.clone(); Arrays.sort(d2);
		int[] index = new int[d2.length]; for(int i=0;i<d2.length;i++)index[i]=i;
		
		//特征向量矩阵的列数
		int cols = v[0].length;
		//完成降序排列
		int M = xmean.getArray().length;
		int N = xmean.getArray()[0].length;
		double[][] vsort = new double[M][cols];// vsort 是一个M*col(注:col一般等于M)阶矩阵，保存的是按降序排列的特征向量,每一列构成一个特征向量
		double[] dsort = new double[M];//dsort 保存的是按降序排列的特征值，是一维行向量
		for(int i=0;i<cols;i++)
		{
			//vsort(:,i) = v(:, index(cols-i+1) );
			for(int j=0;j<M;j++)
				vsort[j][i] = v[j][index[cols-i-1]];
			//dsort(i)   = d1( index(cols-i+1) )
			dsort[i] = d1[index[cols-i-1]];
		}
		//选择90%的能量
		double dsum = sum(dsort);
		double dsum_extract = 0;
	
		int p=-1;
		while(dsum_extract/dsum<0.9)
		{
			p+=1;
			dsum_extract = sum(dsort,0,p);
		}
		int i=0;
		double[][] base = new double[N][p+1];//base是N×p阶矩阵，除以dsort(i)^(1/2)
		//计算特征脸形成的坐标系
		while(i<=p&&dsort[i]>0)
		{
			//base(:,i) = dsort(i)^(-1/2) * xmean' * vsort(:,i);
			double[][] vsort_i = new double[M][1];
			for(int k=0;k<M;k++)
				vsort_i[k][0] = vsort[k][i];
			double[][] tmp = xmean.transpose().times(new Matrix(vsort_i)).times(Math.pow(dsort[i], -0.5)).getArray();
			for(int k=0;k<N;k++)
				base[k][i]=tmp[k][0];
			i+=1;
		}
		//将训练样本对坐标系上进行投影,得到一个 M*p 阶矩阵allcoor
		double[][] allCoor = getAllCoor(base);
		double[][] S = this.unitNormalize(allCoor);
		
		int[] label = new int[allTrainingSamples.size()];
		for(int idx=0;idx<label.length;idx++)
		{
			label[idx] = map.get(records.get(idx));
		}
		writeFile(S,System.getProperty("user.dir")+"/src/resources/texts/TrainX.txt");
		writeFile(label,System.getProperty("user.dir")+"/src/resources/texts/TrainY.txt");
		writeFile(System.getProperty("user.dir")+"/src/resources/texts/NameLabelMap.txt");
		writeFile(base,System.getProperty("user.dir")+"/src/resources/texts/Base.txt");
	}
	public void writeFile(double[][]X,String path)
	{
		FileWriter fileWriter = null;
		try {
			fileWriter = new FileWriter(path);
			for(int i=0;i<X.length;i++)
			{
				for(int j=0;j<X[0].length;j++)
				{
					fileWriter.write(X[i][j]+" ");
				}
				fileWriter.write("\n");
			}
			fileWriter.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally
		{
				try {
					if(fileWriter!=null)
						fileWriter.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}
	}
	public void writeFile(int[]Y,String path)
	{
		FileWriter fileWriter = null;
		try {
			fileWriter = new FileWriter(path);
			for(int i=0;i<Y.length;i++)
			{
				fileWriter.write(Y[i]+"");
				fileWriter.write("\n");
			}
			fileWriter.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally
		{
				try {
					if(fileWriter!=null)
						fileWriter.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}
	}
	public void writeFile(String path)
	{
		FileWriter fileWriter = null;
		try {
			fileWriter = new FileWriter(path);
			for(String key:map.keySet())
			{
				fileWriter.write(key+" "+map.get(key));
				fileWriter.write("\n");
			}
			fileWriter.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally
		{
				try {
					if(fileWriter!=null)
						fileWriter.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}
	}
}
