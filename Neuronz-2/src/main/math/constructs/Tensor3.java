package main.math.constructs;

import main.math.utility.DimensionMismatchException;
import main.math.utility.FloatApplier;
import main.math.utility.FloatOperator;

/**
 * A rank 3 tensor.
 *
 * @author Joe Desmond
 */
public final class Tensor3 extends ElementContainer<Tensor3> {
	
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
	public final Matrix getLayer(int index) {
		return matrices[index];
	}

	@Override
	public Tensor3 elementOperation(final Tensor3 other, final FloatOperator operator) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Tensors must have the same number of layers to perform element operations!");
		}
		
		final Matrix[] result = new Matrix[dimension];
		
		for (int layer = 0; layer < dimension; layer++) {
			result[layer] = matrices[layer].elementOperation(other.matrices[layer], operator);
		}
		
		return new Tensor3(result);
	}

	@Override
	public Tensor3 transform(final FloatApplier transformation) {
		final Matrix[] result = new Matrix[dimension];
		
		for (int layer = 0; layer < dimension; layer++) {
			result[layer] = matrices[layer].transform(transformation);
		}
		
		return new Tensor3(result);
	}
}
