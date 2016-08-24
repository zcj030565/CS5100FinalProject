package edu.neu.coe.opencv.UI;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;
import javax.swing.*;
import java.io.*;
import edu.neu.coe.core.KNNAlgorithm;
import edu.neu.coe.core.LogisticRegressionIII;
import edu.neu.coe.core.Recognition;
import edu.neu.coe.core.SVMAlgorithmII;
import edu.neu.coe.core.ERROR;
public class MainInterface extends JDialog implements ActionListener
{
	private JPanel leftPanel;
	private JPanel rightPanel;
	private JButton trainProcessButton;
	private JButton testProcessButton;
	private JButton knnButton;
	private JButton svmButton;
	private JButton svmModelButton;
	private JButton logisticButton;
	private JButton logisticModelButton;
	private JTextArea errorArea;
	private JScrollPane jsp;
	private List<ERROR> classificationErrors;
	private void initComponents()
	{
		this.leftPanel = new JPanel(new GridLayout(7,1));
		this.rightPanel = new JPanel();
		this.trainProcessButton = new JButton("Training Image Process");
		this.trainProcessButton.addActionListener(this);
		this.testProcessButton = new JButton("Testing Image Process");
		this.testProcessButton.addActionListener(this);
		this.knnButton = new JButton("KNN");
		this.knnButton.addActionListener(this);
		this.svmButton = new JButton("SVM");
		this.svmButton.addActionListener(this);
		this.svmModelButton = new JButton("SVM Model Training");
		this.svmModelButton.addActionListener(this);
		this.logisticButton = new JButton("Logistic Regression");
		this.logisticButton.addActionListener(this);
		this.logisticModelButton = new JButton("Logistic Regression Model Training");
		this.logisticModelButton.addActionListener(this);
		this.errorArea = new JTextArea(35,30);
		this.errorArea.setEditable(false);
		this.jsp = new JScrollPane(errorArea);
		this.classificationErrors = new ArrayList<ERROR>();
		leftPanel.add(trainProcessButton);
		leftPanel.add(testProcessButton);
		leftPanel.add(knnButton);
		leftPanel.add(svmButton);
		leftPanel.add(svmModelButton);
		leftPanel.add(logisticButton);
		leftPanel.add(logisticModelButton);
		rightPanel.add(jsp,BorderLayout.CENTER);
		this.add(leftPanel,BorderLayout.WEST);
		this.add(rightPanel, BorderLayout.CENTER);
		this.setTitle("Face Recognition");
		this.setVisible(true);
		this.setDefaultCloseOperation(this.DISPOSE_ON_CLOSE);
		this.setSize(650, 600);
	}
	public MainInterface()
	{
		initComponents();
	}
	public void updateErrorList()
	{
		this.errorArea.setText("");
		for(ERROR e:classificationErrors)
		{
			String info = "Actual Class Is "+ e.actualClass+", Predicted Class Is "+e.predictedClass+"\n";
			this.errorArea.append(info);
		}
		this.errorArea.append("Total number of Errors is "+classificationErrors.size());
	}

	@Override
	public void actionPerformed(ActionEvent e) 
	{
		// TODO Auto-generated method stub
		if(e.getSource().equals(trainProcessButton))
		{
			Recognition.train(40, 0, 4);
			JOptionPane.showMessageDialog(null, "Finish Processing" ,null, JOptionPane.INFORMATION_MESSAGE);
		}else if(e.getSource().equals(testProcessButton))
		{
			Recognition.test(40, 5,9);
			JOptionPane.showMessageDialog(null, "Finish Processing" ,null, JOptionPane.INFORMATION_MESSAGE);
		}else if(e.getSource().equals(knnButton))
		{
			classificationErrors = KNNAlgorithm.KNN(40);
			updateErrorList();
		}else if(e.getSource().equals(svmButton))
		{
			classificationErrors = SVMAlgorithmII.SVMTest();
			updateErrorList();
		}else if(e.getSource().equals(svmModelButton))
		{
			SVMAlgorithmII.SVM(40, 130, 0.01, 1000,5);
			JOptionPane.showMessageDialog(null, "Finish Training" ,null, JOptionPane.INFORMATION_MESSAGE);
//			System.out.println("训练结束");
		}else if(e.getSource().equals(logisticButton))
		{
			classificationErrors = LogisticRegressionIII.LogisticRegressionTest();
			updateErrorList();
		}else if(e.getSource().equals(logisticModelButton))
		{
			LogisticRegressionIII.LogisticRegression(40, 1000,5);
			JOptionPane.showMessageDialog(null, "Finish Training" ,null, JOptionPane.INFORMATION_MESSAGE);
//			System.out.println("训练结束");
		}
	}
}
