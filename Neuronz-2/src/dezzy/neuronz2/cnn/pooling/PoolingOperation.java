package dezzy.neuronz2.cnn.pooling;

import dezzy.neuronz2.math.constructs.Matrix;

/**
 * A pooling operation; converts a matrix into a scalar. Returns a {@link SliceResult}
 * containg an additional modified input matrix, which is used to calculate the gradient of the
 * error with respect to the original matrix.
 *
 * @author Joe Desmond
 * @see PoolingResult#modifiedInput
 */
@FunctionalInterface
public interface PoolingOperation {
	
	/**
	 * Implementation of max pooling
	 */
	public static final PoolingOperation MAX_POOLING = new MaxPooling();
	
	/**
	 * Condenses the given submatrix to a scalar-matrix pair. Used to implement pooling transformations
	 * for convolutional neural nets. The scalar is the pooled result, and the matrix is a modified slice
	 * of the input matrix, modified for backpropagation in the future. When the gradient matrix is
	 * calculated, the matrices returned from this operation are used as the input matrix.
	 * <p>
	 * If this function implements max pooling, the returned {@linkplain SliceResult#result scalar} would be 
	 * the maximum element in the given matrix, and the returned {@linkplain SliceResult#modifiedInputSlice matrix} would
	 * be a copy of the input matrix, with every element except the max element set to zero. Application of this function
	 * to several submatrices in a larger matrix would yield several of these modified matrices, which would replace the original
	 * submatrices in the larger matrix. Gradients would then be calculated with respect to the larger modified matrix.
	 * 
	 * @param matrix matrix
	 * @return scalar and matrix pair
	 * @see PoolingResult#modifiedInput
	 */
	public SliceResult condense(final Matrix matrix);
}
