package edu.neu.coe.core;

public class Model {
	int preferedLabel;
	int otherLabel;
	double[] weight;
	double bias;
	public Model(int preferedLabel,int otherLabel,double[] weight,double bias)
	{
		this.preferedLabel = preferedLabel;
		this.otherLabel = otherLabel;
		this.weight = weight;
		this.bias = bias;
	}
}
