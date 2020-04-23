package dezzy.neuronz2.math.utility;

import java.io.Serializable;

import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A tensor-valued function. Takes one tensor argument and returns a tensor.
 * Similar to {@link DoubleApplier}.
 * <p>
 * <b>NOTE:</b> A distinction is made between "tensor" here and the 
 * {@link dezzy.neuronz2.math.constructs.Tensor3 Tensor3} class. While the Tensor3
 * class could be used as the type argument, a matrix or vector could be used as well:
 * "tensor" as it is used here refers to its more general, mathematical definition.
 *
 * @author Joe Desmond
 * @param <T> tensor type (can be a vector, matrix, or higher rank tensor)
 */
@FunctionalInterface
public interface TensorApplier<T extends ElementContainer<T>> extends Serializable {
	
	/**
	 * Accepts a tensor argument and returns a tensor with the same rank.
	 * 
	 * @param x input tensor
	 * @return output tensor
	 */
	T apply(final T x);
}
