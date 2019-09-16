package main.math.constructs;

import java.io.Serializable;

import main.math.utility.DoubleApplier;

/**
 * Contains a function and its partial derivative with respect to the original function's input.
 *
 * @author Joe Desmond
 */
public final class FuncDerivPair implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8265927003886644864L;

	/**
	 * The sigmoid activation function
	 */
	public static final FuncDerivPair SIGMOID = new FuncDerivPair(
			x -> 1.0f/(double)(1.0f + Math.exp(-x)),
			sigmoid -> sigmoid * (1 - sigmoid)
	);
	
	/**
	 * Rectified Linear Unit activation function
	 */
	public static final FuncDerivPair RELU = new FuncDerivPair(
			x -> (double)Math.max(0, x),
			relu -> (relu == 0) ? 0 : 1
	);
	
	/**
	 * The activation function
	 */
	public final DoubleApplier function;
	
	/**
	 * The partial derivative of the function with respect to the original function's input
	 */
	public final DoubleApplier partialDerivative;
	
	/**
	 * Creates a new FuncDerivPair with the given function and partial derivative.
	 * 
	 * @param _function original function
	 * @param _partialDerivative partial derivative of <code>_function</code>
	 */
	public FuncDerivPair(final DoubleApplier _function, final DoubleApplier _partialDerivative) {
		function = _function;
		partialDerivative = _partialDerivative;
	}
}
