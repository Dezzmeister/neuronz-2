package dezzy.neuronz2.math.constructs.shape;

import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * The shape of a rank 3 tensor. Contains the number of layers in the tensor as well as
 * the number of rows and columns in each tensor.
 *
 * @author Joe Desmond
 */
public final class Tensor3Shape extends Shape<Tensor3> {
	
	/**
	 * The number of layers (matrices) in the tensor 
	 */
	public final int layers;
	
	/**
	 * The number of rows in each layer
	 */
	public final int rows;
	
	/**
	 * The number of columns in each layer
	 */
	public final int cols;
	
	/**
	 * Constructs an object containing the shape of a tensor. A tensor with this shape
	 * can be generated with {@link #generate(IndexedGenerator)}.
	 * 
	 * @param _layers number of layers (matrices) in the tensor
	 * @param _rows number of rows in the tensor
	 * @param _cols number of columns in the tensor
	 */
	public Tensor3Shape(final int _layers, final int _rows, final int _cols) {
		layers = _layers;
		rows = _rows;
		cols = _cols;
	}
	
	@Override
	public Tensor3 generate(final IndexedGenerator generator) {
		return Tensor3.generate(generator, layers, rows, cols);
	}
	
	@Override
	public String toString() {
		return "(" + layers + ", " + rows + ", " + cols + ")";
	}
}
