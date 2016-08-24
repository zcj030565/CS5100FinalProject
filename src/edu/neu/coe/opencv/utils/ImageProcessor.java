package edu.neu.coe.opencv.utils;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.*;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class ImageProcessor 
{
	public static BufferedImage toBufferedImage(Mat image)
	{
		int type = BufferedImage.TYPE_BYTE_GRAY;
		if(image.channels()>1)
			type = BufferedImage.TYPE_3BYTE_BGR;
		int bufferSize = image.channels()*image.cols()*image.rows();
		byte[] buffer = new byte[bufferSize];
		image.get(0, 0, buffer);
		BufferedImage targetImage = new BufferedImage(image.cols(),image.rows(),type);
		final byte[] targetPixels = ((DataBufferByte) targetImage.getRaster().getDataBuffer()).getData();
		System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);  
		return targetImage;
	}
	public static Mat getRange(Mat original,Rect rect)
	{
		int x = rect.x;
		int y = rect.y;
		int width = rect.width;
		int height = rect.height;
		int bufferSize = original.channels()*original.cols()*original.rows();
		Mat rowRange = original.rowRange(new Range(y,y+height-1));
		Mat colRange = rowRange.colRange(new Range(x,x+width-1));
		return colRange;
	}
	public static Mat normalizeTarget(Mat image)
	{
		int startX = (image.cols()-140)/2;
		int startY = (image.rows()-140)/2;
		Mat colRange = image.colRange(startX, startX+140);
		Mat rowRange = colRange.rowRange(startY, startY+140);
		return rowRange;
	}
}

