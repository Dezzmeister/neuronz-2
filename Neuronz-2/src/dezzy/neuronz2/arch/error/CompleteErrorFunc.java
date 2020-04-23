package dezzy.neuronz2.arch.error;

import java.io.Serializable;

import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * An error/cost function and its derivative with respect to the actual network output.
 *
 * @author Joe Desmond
 * @param <T> tensor type (vector, matrix, etc.)
 */
public class CompleteErrorFunc<T extends ElementContainer<T>> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5980746604513514547L;
	
	/**
	 * The error/cost function
	 */
	public final TensorErrorFunc<T> errorFunction;
	
	/**
	 * The derivative of the error/cost function with respect to the actual network output
	 */
	public final TensorErrorFuncDeriv<T> errorFunctionDerivative;
	
	/**
	 * Constructs a complete error function given the actual error function and its derivative.
	 * 
	 * @param _errorFunction the error function
	 * @param _errorFunctionDerivative derivative of the error function with respect to the actual network output
	 */
	public CompleteErrorFunc(final TensorErrorFunc<T> _errorFunction, final TensorErrorFuncDeriv<T> _errorFunctionDerivative) {
		errorFunction = _errorFunction;
		errorFunctionDerivative = _errorFunctionDerivative;
	}
}
