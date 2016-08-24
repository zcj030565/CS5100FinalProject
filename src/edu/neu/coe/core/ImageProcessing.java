package edu.neu.coe.core;
import java.io.*;

import javax.imageio.ImageIO;
import java.util.*;
import java.awt.image.*;
import Jama.*;
public class ImageProcessing 
{
	/**
	 * 
	 * @param samples
	 * @return the average value of each column
	 */
	public static double[] meanImage(List<double[]> samples)
	{
		int size = samples.get(0).length;
		double[] result = new double[size];
		for(int i=0;i<size;i++)
		{
			double sum = 0;
			for(int j=0;j<samples.size();j++)
			{
				sum += samples.get(j)[i];
			}
			result[i] = sum/samples.size();
		}
		return result;
	}
	/**
	 * 
	 * @param mean
	 * @return the matrix of each image after subtracting the average
	 */
	public static Matrix subAverage(double[] mean,List<double[]> samples)
	{
		List<double[]> list = new ArrayList<double[]>();
		for(int i=0;i<samples.size();i++)
		{
			double[] sp = samples.get(i);
			list.add(Utils.vectorSubVector(sp, mean));
		}
		double[][] result = new double[list.size()][list.get(0).length];
		for(int i=0;i<list.size();i++)
			result[i] = list.get(i);
		return new Matrix(result);
	}

	/**
	 * 
	 * @param path
	 * @return gray scale of the file
	 */
	public static double[][] image2Array(String path)
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
						//&0xff 1111 1111 Get last 8 digit
						//It's a gray image, so R,G,B is the same. Keep last 8 digit only.2^24->2^8
						map[y][x]=rgb[2];// The number of map is 112*92.
						//The difference between expression of image and array. x<->y
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
	/**
	 * 
	 * @param samples
	 * @param base
	 * @return the matrix after being projected to the given coordinate
	 */
	public static double[][] projectImageToCoordinate(List<double[]> samples,double[][] base)
	{
		double[][] matrix = new double[samples.size()][];
		for(int i=0;i<samples.size();i++)
		{
			matrix[i] = samples.get(i);
		}
		Matrix res = (new Matrix(matrix)).times(new Matrix(base));
		return res.getArray();
	}

}
