package dezzy.neuronz2.math.constructs;

import java.io.Serializable;

import dezzy.neuronz2.math.utility.DimensionMismatchException;
import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;



/**
 * A rank 3 tensor.
 *
 * @author Joe Desmond
 */
public final class Tensor3 extends ElementContainer<Tensor3> implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -5804689492753369823L;

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
	public Tensor3(final Matrix ... _matrices) {
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
	
	/**
	 * Sets the layer at the given index to the given matrix.
	 * 
	 * @param index index of layer to be replaced
	 * @param layer layer to replace previous layer at <code>index</code>
	 */
	public final void setLayer(int index, final Matrix layer) {
		matrices[index] = layer;
	}

	@Override
	public Tensor3 elementOperation(final Tensor3 other, final DoubleOperator operator) {
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
	public Tensor3 transform(final DoubleApplier transformation) {
		final Matrix[] result = new Matrix[dimension];
		
		for (int layer = 0; layer < dimension; layer++) {
			result[layer] = matrices[layer].transform(transformation);
		}
		
		return new Tensor3(result);
	}
	
	@Override
	public final String toString() {
		String out = "[";
		
		for (int layer = 0; layer < dimension; layer++) {
			out += matrices[layer].toString();
			
			if (layer != dimension - 1) {
				out += "\n\n";
			} else {
				out += "]";
			}
		}
		
		return out;
	}
}