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
}
