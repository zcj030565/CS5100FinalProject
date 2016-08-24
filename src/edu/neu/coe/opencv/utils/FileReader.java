package edu.neu.coe.opencv.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FileReader {
	public static double[][] getFile(String path)
	{
		List<List<Double>> baseList = new ArrayList<List<Double>>();
		File file = new File(path);
		FileInputStream fis = null;
		InputStreamReader isr = null;
		BufferedReader br = null;
		if(file.exists())
		{
			try
			{
				fis = new FileInputStream(file);
				isr = new InputStreamReader(fis);
				br = new BufferedReader(isr);
				String lineTxt ="";
				while((lineTxt=br.readLine())!=null)
				{
					String[] lineArr = lineTxt.trim().split(" ");
					List<Double> row = new ArrayList<Double>();
					for(String elem:lineArr)
					{
						row.add(Double.parseDouble(elem));
					}
					baseList.add(row);
				}
				int rows = baseList.size();
				int cols = baseList.get(0).size();
				double[][] array = new double[rows][cols];
				for(int i=0;i<rows;i++)
				{
					for(int j=0;j<cols;j++)
					{
						array[i][j] = baseList.get(i).get(j);
					}
				}
				return array;
			}catch(Exception e)
			{
				e.printStackTrace();
			}
			
		}
		return null;
	}
	public static Map<Integer,String> readMapFile(String path)
	{
		Map<Integer,String> map = new HashMap<Integer,String>();
		File file = new File(path);
		FileInputStream fis = null;
		InputStreamReader isr = null;
		BufferedReader br = null;
		String lineTxt = "";
		try
		{
			fis =new FileInputStream(file);
			isr = new InputStreamReader(fis);
			br = new BufferedReader(isr);
			while((lineTxt=br.readLine())!=null)
			{
				String[] lineArr = lineTxt.trim().split(" ");
				int key = Integer.parseInt(lineArr[1]);
				String value = lineArr[0];
				map.put(key, value);
			}
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		return map;
	}
}
