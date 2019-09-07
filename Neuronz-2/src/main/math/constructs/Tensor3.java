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
	
	/**
	 * Returns the matrix at the given index. Does not check if <code>index</code> is within acceptable bounds.
	 * 
	 * @param index must be greater than or equal to 0 and less than {@link Tensor3#dimension}
	 * @return the Matrix at <code>index</code>
	 */
	public final Matrix getMatrixAt(int index) {
		return matrices[index];
	}
}
