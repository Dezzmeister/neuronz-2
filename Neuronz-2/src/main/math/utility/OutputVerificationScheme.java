package main.math.utility;

import main.math.constructs.Vector;

/**
 * Defines a scheme to evaluate the success of one forward pass of a neural network.
 *
 * @author Joe Desmond
 */
public interface OutputVerificationScheme {
	
	/**
	 * Returns true if the network's actual output is close enough to the expected output.
	 * 
	 * @param actualOutput actual output from one forward pass
	 * @param expectedOutput expected output from one forward pass
	 * @return true if the actual output is close enough to the expected output, as defined by some evaluation logic
	 */
	boolean isSuccess(final Vector actualOutput, final Vector expectedOutput);
	
	
	/**
	 * An output verification scheme that finds the neuron producing the greatest output, and checks to see
	 * if the expected output for that neuron is 1.
	 */
	public static final OutputVerificationScheme greatestOutputMetric = (actual, expected) -> {
		int greatestIndex = 0;
		double greatestValue = 0;
		
		for (int i = 0; i < actual.dimension; i++) {
			double currentValue = actual.get(i);
			
			if (currentValue > greatestValue) {
				greatestValue = currentValue;
				greatestIndex = i;
			}
		}
		//System.out.println(greatestValue + "\t" + greatestIndex + "\t" + expected.get(greatestIndex));
		
		return expected.get(greatestIndex) == 1;
	};
}
