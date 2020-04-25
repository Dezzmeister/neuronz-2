package dezzy.neuronz2.math.constructs.shape;

import dezzy.neuronz2.math.constructs.Tensor4;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * The shape of a rank-4 tensor. Contains the number of rank-3 tensors in the tensor, 
 * the number of layers (matrices) in each rank-3 tensor, and the number of rows and columns in each layer.
 *
 * @author Joe Desmond
 */
public final class Tensor4Shape extends Shape<Tensor4> {
	
	/**
	 * Number of rank-3 tensors in this tensor
	 */
	public final int tensors;
	
	/**
	 * Number of layers (matrices) in each rank-3 tensor
	 */
	public final int layers;
	
	/**
	 * Number of rows in each layer
	 */
	public final int rows;
	
	/**
	 * Number of columns in each layer
	 */
	public final int cols;
	
	/**
	 * Creates an object containing the shape of a rank-4 tensor. A tensor with this shape
	 * can be generated with {@link #generate(IndexedGenerator)}.
	 * 
	 * @param _tensors number of rank-3 tensors in this tensor
	 * @param _layers number of layers (matrices) in each rank-3 tensor
	 * @param _rows number of rows in each layer
	 * @param _cols number of columns in each layer
	 */
	public Tensor4Shape(final int _tensors, final int _layers, final int _rows, final int _cols) {
		tensors = _tensors;
		layers = _layers;
		rows = _rows;
		cols = _cols;
	}
	
	@Override
	public Tensor4 generate(IndexedGenerator generator) {
		return Tensor4.generate(generator, tensors, layers, rows, cols);
	}
	
	@Override
	public String toString() {
		return "(" + tensors + ", " + layers + ", " + rows + ", " + cols + ")";
	}
}
