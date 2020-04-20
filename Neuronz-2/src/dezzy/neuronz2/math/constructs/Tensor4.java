package dezzy.neuronz2.math.constructs;

import java.io.Serializable;

import dezzy.neuronz2.math.utility.DimensionMismatchException;
import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;

/**
 * A rank 4 tensor.
 *
 * @author Joe Desmond
 */
public class Tensor4 extends ElementContainer<Tensor4> implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3049166175032426931L;
	
	/**
	 * A rank 4 tensor can be represented as a series of rank 3 tensors
	 */
	private final Tensor3[] tensors;
	
	/**
	 * Number of rank 3 tensors in this tensor
	 */
	public final int dimension;
	
	/**
	 * Constructs a rank 4 tensor from the given rank 3 tensors.
	 * 
	 * @param _tensors rank 3 tensors
	 */
	public Tensor4(final Tensor3 ... _tensors) {
		tensors = _tensors;
		dimension = tensors.length;
	}
	
	/**
	 * Returns the tensor at the given index. Does not check if <code>index</code> is within
	 * acceptable bounds.
	 * 
	 * @param index index of the tensor
	 * @return the {@link Tensor3} at <code>index</code>
	 */
	public final Tensor3 getTensor(final int index) {
		return tensors[index];
	}
	
	/**
	 * Sets the tensor at the given index to the given tensor.
	 * 
	 * @param index index of tensor to be replaced
	 * @param tensor tensor to replace previous tensor at <code>index</code>
	 */
	public void setTensor(final int index, final Tensor3 tensor) {
		tensors[index] = tensor;
	}
	
	@Override
	public Tensor4 elementOperation(final Tensor4 other, final DoubleOperator operator) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Tensors must have the same number of layers to perform element operations!");
		}
		
		final Tensor3[] result = new Tensor3[dimension];
		
		for (int i = 0; i < dimension; i++) {
			result[i] = tensors[i].elementOperation(other.tensors[i], operator);
		}
		
		return new Tensor4(result);
	}

	@Override
	public Tensor4 transform(DoubleApplier transformation) {
		final Tensor3[] result = new Tensor3[dimension];
		
		for (int i = 0; i < dimension; i++) {
			result[i] = tensors[i].transform(transformation);
		}
		
		return new Tensor4(result);
	}
	
}
