package dezzy.neuronz2.cnn.pooling;

import java.io.Serializable;

import dezzy.neuronz2.math.constructs.Matrix;

/**
 * A pooling operation; converts a matrix into a scalar.
 *
 * @author Joe Desmond
 * @see PoolingResult#modifiedInput
 */
public interface PoolingOperation extends Serializable {
	
	/**
	 * Implementation of max pooling
	 */
	public static final PoolingOperation MAX_POOLING = new MaxPooling();
	
	/**
	 * Condenses the given submatrix to a scalar. Operates on a single pooling window.
	 * 
	 * @param matrix matrix
	 * @return scalar
	 */
	public double condense(final Matrix matrix);
	
	/**
	 * Calculates the partial derivative of the error with respect to the input to this pooling
	 * operation. The derivative matrix is the same size as the output of {@link #condense()},
	 * and the returned matrix is the same size as the input to {@link #condense()}.
	 * <p>
	 * Unlike {@link #condense()}, this function operates on the entire input and derivative
	 * matrices ({@link #condense()} operates only on windows of the input matrix). 
	 * 
	 * @param latestInput latest input to {@link #condense()}
	 * @param derivative partial derivative of the error of a network with respect to the output of 
	 * 			{@link #condense(Matrix) condense(latestInput)}
	 * @param windowRows number of rows in pooling window
	 * @param windowCols number of columns in pooling window
	 * @param rowStride row stride
	 * @param colStride column stride
	 * @return partial derivative of the error of a network with respect to <code>latestInput</code>
	 * 			(use the chain rule with <code>derivative</code>)
	 */
	public Matrix backprop(final Matrix latestInput, final Matrix derivative, final int windowRows, final int windowCols, final int rowStride, final int colStride);
}
