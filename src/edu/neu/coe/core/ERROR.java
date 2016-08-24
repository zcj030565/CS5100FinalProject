package edu.neu.coe.core;

public class ERROR 
{
	public int actualClass;
	public int predictedClass;
	public ERROR(int actualClass,int predictedClass)
	{
		this.actualClass = actualClass;
		this.predictedClass = predictedClass;
	}
}
