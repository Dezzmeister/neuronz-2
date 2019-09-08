package main.math.constructs;

import main.math.utility.FloatApplier;

/**
 * Contains a function and its partial derivative with respect to the original function's input.
 *
 * @author Joe Desmond
 */
public final class FuncDerivPair {
	
	/**
	 * The sigmoid activation function
	 */
	public static final FuncDerivPair SIGMOID = new FuncDerivPair(
			x -> 1.0f/(float)(1.0f + Math.exp(-x)),
			sigmoid -> sigmoid * (1 - sigmoid)
	);
	
	/**
	 * Rectified Linear Unit activation function
	 */
	public static final FuncDerivPair RELU = new FuncDerivPair(
			x -> (float)Math.max(0, x),
			relu -> (relu == 0) ? 0 : 1
	);
	
	/**
	 * The activation function
	 */
	public final FloatApplier function;
	
	/**
	 * The partial derivative of the function with respect to the original function's input
	 */
	public final FloatApplier partialDerivative;
	
	/**
	 * Creates a new FuncDerivPair with the given function and partial derivative.
	 * 
	 * @param _function original function
	 * @param _partialDerivative partial derivative of <code>_function</code>
	 */
	public FuncDerivPair(final FloatApplier _function, final FloatApplier _partialDerivative) {
		function = _function;
		partialDerivative = _partialDerivative;
	}
}
