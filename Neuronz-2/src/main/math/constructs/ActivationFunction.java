package main.math.constructs;

import main.math.utility.FloatApplier;

/**
 * Contains an activation function and its partial derivative with respect to its input.
 *
 * @author Joe Desmond
 */
public final class ActivationFunction {
	
	/**
	 * The sigmoid activation function
	 */
	public static final ActivationFunction SIGMOID = new ActivationFunction(
			x -> 1.0f/(float)(1.0f + Math.exp(-x)),
			sigmoid -> sigmoid * (1 - sigmoid)
	);
	
	/**
	 * Rectified Linear Unit activation function
	 */
	public static final ActivationFunction RELU = new ActivationFunction(
			x -> (float)Math.max(0, x),
			relu -> (relu == 0) ? 0 : 1
	);
	
	/**
	 * The activation function
	 */
	public final FloatApplier function;
	
	/**
	 * The partial derivative of the activation function with respect to its input
	 */
	public final FloatApplier partialDerivative;
	
	public ActivationFunction(final FloatApplier _function, final FloatApplier _partialDerivative) {
		function = _function;
		partialDerivative = _partialDerivative;
	}
}
