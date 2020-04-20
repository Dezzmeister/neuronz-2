package dezzy.neuronz2.cnn.pooling;

import dezzy.neuronz2.math.constructs.Matrix;

/**
 * The output of a pooling operation applied to a single slice of a larger matrix.
 * Contains the actual scalar input and a slice of a modified input matrix
 * to be used when calculating gradients.
 *
 * @author Joe Desmond
 * @see PoolingOperation#condense(Matrix)
 * @see PoolingResult#modifiedInput
 */
public class SliceResult {
	
	/**
	 * The result of applying a pooling operation to a slice
	 */
	public final double result;
	
	/**
	 * The modified slice of the input matrix (see {@link PoolingResult#modifiedInput})
	 */
	public final Matrix modifiedInputSlice;
	
	/**
	 * Constructs a SliceResult with the given pooling result and modified input
	 * matrix slice.
	 * 
	 * @param _result pooling result
	 * @param _modifiedInputSlice modified input matrix slice
	 */
	public SliceResult(final double _result, final Matrix _modifiedInputSlice) {
		result = _result;
		modifiedInputSlice = _modifiedInputSlice;
	}
}
