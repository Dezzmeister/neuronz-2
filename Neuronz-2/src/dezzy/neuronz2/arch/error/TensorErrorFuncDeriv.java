package dezzy.neuronz2.arch.error;

import java.io.Serializable;

import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * The derivative of a {@link TensorErrorFunc}. This is used to define the derivatives
 * of error functions; a function of this type is paired with its antiderivative in {@link CompleteErrorFunc}.
 * <p>
 * These functions are generalized to operate on tensors of any rank. Although they usually operate on vectors,
 * it may be desired that they operate on matrices instead, or some other higher-order tensor.
 *
 * @author Joe Desmond
 * @param <T> tensor type (vector, matrix, etc.)
 */
@FunctionalInterface
public interface TensorErrorFuncDeriv<T extends ElementContainer<T>> extends Serializable {
	
	/**
	 * Calculates the partial derivative of some {@link TensorErrorFunc DualTensorCondenser's}
	 * output with respect to the actual input. 
	 * 
	 * @param expected expected network output
	 * @param actual actual network output
	 * @param error calculated error
	 * @return (partial) derivative of the error with respect to the actual output
	 */
	T calculate(final T expected, final T actual, final double error);
}
