package edu.neu.coe.opencv.UI;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;
import javax.swing.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import edu.neu.coe.opencv.utils.ImageProcessor;
public class RegisterGUI extends JDialog implements ActionListener
{
	private JLabel imageView;
	private JPanel imagePanel;
	private JLabel name;
	private JTextField nameText;
	
	private JPanel infoPanel;
	private JButton confirm;
	private JButton cancel;
	private JPanel buttonPanel;
	private BufferedImage finalImage;
	private Mat savedImage = new Mat();
	public RegisterGUI(BufferedImage image)
	{
		this.finalImage=image.getSubimage((image.getWidth()-120)/2, (image.getHeight()-120)/2, 120, 120);
		initGUI();
		setImage(image);
	}
	public RegisterGUI(Mat image)
	{
		savedImage = ImageProcessor.normalizeTarget(image);
		finalImage = ImageProcessor.toBufferedImage(savedImage);
		initGUI();
		setImage(finalImage);
	}
	private void initGUI()
	{
		imageView = new JLabel();
		imagePanel = new JPanel();
		imagePanel.add(imageView);
		imagePanel.setPreferredSize(new Dimension(400,400));
		name = new JLabel("Name",JLabel.CENTER);
		nameText = new JTextField();
		nameText.setPreferredSize(new Dimension(300,30));
		infoPanel = new JPanel();
		infoPanel.add(name);
		infoPanel.add(nameText);
		confirm = new JButton("Confirm");
		confirm.addActionListener(this);
		confirm.setActionCommand("confirm");
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		cancel.setActionCommand("cancel");
		buttonPanel = new JPanel();
		buttonPanel.add(confirm);
		buttonPanel.add(cancel);
		this.add(imagePanel,BorderLayout.NORTH);
		this.add(infoPanel,BorderLayout.CENTER);
		this.add(buttonPanel,BorderLayout.SOUTH);
		this.setVisible(true);
		this.setSize(480, 500);
		this.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		
	}
	public void setImage(Image image)
	{
		imageView.setIcon(new ImageIcon(image));
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO Auto-generated method stub
		if(e.getActionCommand().equals("confirm"))
		{
			if(!nameText.getText().isEmpty())
			{
				String filePath = System.getProperty("user.dir")+"/src/images/"+nameText.getText();
				int index =0;
				while(new File(filePath+"_"+index+".bmp").exists())
				{
					index++;
				}
				filePath = filePath+"_"+(index%10)+".bmp";
				try 
				{
					Imgproc.cvtColor(savedImage, savedImage, Imgproc.COLOR_BGR2GRAY);
					Imgcodecs.imwrite(filePath, savedImage);
				} catch (Exception e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}else
			{
				JOptionPane jop = new JOptionPane();
				jop.showMessageDialog(this, "Please enter name");
			}	
		}
		this.dispose();
	}
}
