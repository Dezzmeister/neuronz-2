package main.network;

import main.math.constructs.Matrix;
import main.math.constructs.Vector;

/**
 * Contains general mathematical functions for neural networks.
 *
 * @author Joe Desmond
 */
public final class NetworkFunctions {
	
	/**
	 * Computes the output vector for a layer of neurons given a weight matrix, the previous activation vector, and biases.
	 * 
	 * @param weights weight matrix
	 * @param prevActivation previous activation vector
	 * @param biases biases
	 * @return the output vector, <code>(weights * prevActivation + biases)</code>
	 */
	public static final Vector computeOutputVector(final Matrix weights, final Vector prevActivation, final Vector biases) {
		return weights.multiply(prevActivation).plus(biases);
	}
}
