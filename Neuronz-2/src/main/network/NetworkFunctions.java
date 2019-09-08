package main.network;

import main.math.constructs.Matrix;
import main.math.constructs.Vector;
import main.math.utility.FloatApplier;
import main.math.utility.FloatOperator;

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
	 * @param activationFunction activation function to squash the outputs
	 * @return the output vector, <code>(weights * prevActivation + biases)</code>
	 */
	public static final Vector computeOutputVector(final Matrix weights, final Vector prevActivation, final Vector biases, final FloatApplier activationFunction) {
		return weights.multiply(prevActivation).plus(biases).transform(activationFunction);
	}
	
	/**
	 * Computes the output vector for a layer of neurons given a weight matrix and the previous activation vector. This version of 
	 * {@link NetworkFunctions#computeOutputVector} should be used when the last column of the weight matrix contains the biases
	 * and the last component of the previous activation vector is 1.
	 * 
	 * @param weights weight matrix (potentially with biases in last column)
	 * @param prevActivation previous activation vector (potentially with 1 as last component)
	 * @param activationFunction activation function to squash the outputs
	 * @return the output vector, <code>(weights * prevActivation)</code>
	 */
	public static final Vector computeOutputVector(final Matrix weights, final Vector prevActivation, final FloatApplier activationFunction) {
		return weights.multiply(prevActivation).transform(activationFunction);
	}
	
	/**
	 * The mean squared error function as a FloatOperator, multiplied by 1/2 to cancel the squared term when differentiating.
	 */
	private static final FloatOperator MSE = (ideal, actual) -> 0.5f * (ideal - actual) * (ideal - actual);
	
	/**
	 * Computes the total mean squared error given an ideal vector and an actual vector. Both vectors must have the same dimension.
	 * 
	 * @param ideal desired values
	 * @param actual actual values
	 * @return sum of MSE calculated for each pair of elements in (ideal, actual)
	 */
	public static final float computeTotalMSE(final Vector ideal, final Vector actual) {
		final Vector mseVector = ideal.elementOperation(actual, MSE);
		
		float sum = 0;
		for (int i = 0; i < mseVector.dimension; i++) {
			sum += mseVector.get(i);
		}
		
		return sum;
	}
}
