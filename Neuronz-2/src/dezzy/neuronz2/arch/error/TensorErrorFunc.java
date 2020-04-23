package dezzy.neuronz2.arch.error;

import java.io.Serializable;

import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A function accepting two tensors and returning a double. Used to implement cost functions,
 * which take an expected tensor and an actual tensor and return the error.
 *
 * @author Joe Desmond
 * @param <T> tensor type (vector, matrix, etc.)
 */
@FunctionalInterface
public interface TensorErrorFunc<T extends ElementContainer<T>> extends Serializable {
	
	/**
	 * Accepts two tensors and returns a double corresponding to the error between the
	 * expected and actual tensors.
	 * 
	 * @param expected expected network output
	 * @param actual actual network output
	 * @return error
	 */
	double condense(final T expected, final T actual);
}
