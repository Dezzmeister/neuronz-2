package main.math.constructs;

/**
 * A rank 3 tensor.
 *
 * @author Joe Desmond
 */
public final class Tensor3 {
	
	/**
	 * Values of the tensor
	 */
	private final Matrix[] matrices;
	
	/**
	 * The number of matrices in this tensor
	 */
	public final int dimension;
	
	/**
	 * Creates a rank 3 tensor from the given matrices.
	 * 
	 * @param _matrices values of the tensor
	 */
	public Tensor3(final Matrix[] _matrices) {
		matrices = _matrices;
		dimension = matrices.length;
	}
	
	 
}
