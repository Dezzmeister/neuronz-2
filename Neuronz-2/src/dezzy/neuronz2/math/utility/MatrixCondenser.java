package dezzy.neuronz2.math.utility;

import dezzy.neuronz2.math.constructs.Matrix;

/**
 * An operation that converts a matrix into a scalar. 
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface MatrixCondenser {
	
	/**
	 * Condenses this {@link Matrix} to a scalar (double). Used to implement pooling transformations
	 * for convolutional neural nets.
	 * 
	 * @param matrix matrix
	 * @return scalar
	 */
	public double condense(final Matrix matrix);
}
