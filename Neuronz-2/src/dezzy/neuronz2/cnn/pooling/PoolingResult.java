package dezzy.neuronz2.cnn.pooling;

import dezzy.neuronz2.math.constructs.Matrix;

/**
 * The result of a pooling operation: a matrix containing the pooled results and a modified input
 * matrix, used to calculate gradients for backpropagation.
 *
 * @author Joe Desmond
 */
public class PoolingResult {
	
	/**
	 * The result of the pooling operation
	 */
	public final Matrix result;
	
	/**
	 * The original input to the pooling operation, modified to be used to calculate gradients
	 * during backpropagation.
	 * <p>
	 * <b>Example:</b> When calculating the gradient of the error with respect to the input matrix (really a tensor)
	 * for a layer, the pooling operation used must be taken into account. If max pooling was used, then the error
	 * depends only on certain values in the input matrix, since those values were selected for the output.
	 * In order to accurately calculate the gradient, a modified input matrix must be used. In this modified matrix,
	 * the values that were not part of the pooling operation's output must be set to zero. For max pooling, the modified
	 * matrix is just a copy of the original input matrix, with every non-output element set to zero.
	 */
	public final Matrix modifiedInput;
	
	/**
	 * Constructs a new pooling result from the given pooling output matrix and modified
	 * input matrix.
	 * 
	 * @param _result pooling output
	 * @param _modifiedInput modified input matrix (for backpropagation)
	 * @see #modifiedInput
	 */
	public PoolingResult(final Matrix _result, final Matrix _modifiedInput) {
		result = _result;
		modifiedInput = _modifiedInput;
	}
}
