package dezzy.neuronz2.math.constructs;

import dezzy.neuronz2.math.constructs.shape.Tensor3Shape;
import dezzy.neuronz2.math.utility.DimensionMismatchException;
import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;
import dezzy.neuronz2.math.utility.IndexedGenerator;



/**
 * A rank 3 tensor.
 *
 * @author Joe Desmond
 */
public final class Tensor3 extends ElementContainer<Tensor3> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -5804689492753369823L;

	/**
	 * Layers of the tensor
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
	 * Returns the shape of this tensor (number of layers, rows, and columns).
	 * 
	 * @return the shape of this tensor
	 */
	public Tensor3Shape shape() {
		return new Tensor3Shape(dimension, matrices[0].rows, matrices[0].cols);
	}
	
	/**
	 * Creates a rank 3 tensor from the given array of values. The array must be completely uniform;
	 * an exception will be thrown if an irregular array is given. This means that every 2D array in
	 * <code>values</code> must share the same dimensions (rows must be the same for all, and columns must be
	 * the same for all).
	 * 
	 * @param values 3D array, or an array of 2D arrays
	 */
	public Tensor3(final double[][][] values) {
		dimension = values.length;
		matrices = new Matrix[dimension];
		
		for (int i = 0; i < values.length; i++) {
			if (i == 0) {
				matrices[0] = new Matrix(values[i]);
			} else {
				matrices[i] = new Matrix(values[i]);
				
				if (!matrices[i].isSameDimensionsAs(matrices[0])) {
					throw new DimensionMismatchException("Tensor must be uniform; array cannot be irregular!");
				}
			}
		}
	}
	
	/**
	 * Generates a tensor with the specified size using the given generator function. Calls
	 * {@link IndexedGenerator#generate(int...) generator.generate(layer, row, col)} with every element's layer, row, and column
	 * to obtain the values of the tensor.
	 * 
	 * @param generator generator function, used to generate components of the tensor
	 * @param layers number of layers (matrices) in the tensor
	 * @param rows number of rows in each layer
	 * @param cols number of columns in each row
	 * @return a new rank 3 tensor
	 */
	public static Tensor3 generate(final IndexedGenerator generator, final int layers, final int rows, final int cols) {
		final double[][][] out = new double[layers][rows][cols];
		
		for (int layer = 0; layer < layers; layer++) {
			for (int row = 0; row < rows; row++) {
				for (int col = 0; col < cols; col++) {
					out[layer][row][col] = generator.generate(layer, row, col);
				}
			}
		}
		
		return new Tensor3(out);
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
	
	/**
	 * The dot product for vectors, extended to tensors. The given tensor must have the same dimensions as this tensor.
	 * Just like the vector dot product, element-wise multiplication is performed on the tensors, and every element of the result
	 * is summed to produce the tensor dot product.
	 * <p>
	 * <b>NOTE:</b> This method actually calculates the tensor product by summing the {@linkplain Matrix#frobenius(Matrix) frobenius products}
	 * of all the corresponding layers.
	 * 
	 * @param other other tensor
	 * @return tensor dot product of this and the other tensor
	 */
	public final double tensorDot(final Tensor3 other) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Tensors must be the same size to calculate tensor dot product!");
		}
		
		double sum = 0;
		
		for (int i = 0; i < matrices.length; i++) {
			sum += matrices[i].frobenius(other.matrices[i]);
		}
		
		return sum;
	}
	
	/**
	 * Gets a subtensor within this tensor. This refers to a smaller tensor that would be located somewhere inside this
	 * tensor. Works just like {@link Matrix#submatrix(int, int, int, int) Matrix.submatrix()}, but extended to rank 3 tensors.
	 * 
	 * @param row starting row (inclusive) within this tensor
	 * @param col starting column (inclusive) within this tensor
	 * @param layer starting layer (inclusive) within this tensor
	 * @param subRows number of rows in the subtensor
	 * @param subCols number of columns in the subtensor
	 * @param subLayers number of layers in the subtensor
	 * @return a subtensor with size <code>[subRows][subCols][subLayers]</code>
	 */
	final Tensor3 subtensor(final int row, final int col, final int layer, final int subRows, final int subCols, final int subLayers) {
		final Matrix[] out = new Matrix[subLayers];
		
		for (int i = layer; i < layer + subLayers; i++) {
			out[i - layer] = matrices[i].submatrix(row, col, subRows, subCols);
		}
		
		return new Tensor3(out);
	}
	
	/**
	 * Convolves this tensor with a kernel and applies a modifier function to each resulting element. This is technically the
	 * cross-correlation operation because the kernel is never flipped.
	 * 
	 * @param kernel kernel (another tensor)
	 * @param stride kernel stride
	 * @param modifier modifier function
	 * @return convolution of this tensor and the kernel, with modifier function applied
	 */
	public final Tensor3 convolve(final Tensor3 kernel, final int stride, final DoubleApplier modifier) {
		final double[][][] out = new double[((dimension - kernel.dimension) / stride) + 1][((matrices[0].rows - kernel.matrices[0].rows) / stride) + 1][((matrices[0].cols - kernel.matrices[0].cols) / stride) + 1];
		
		int depthIndex = 0;
		for (int depth = 0; depth < dimension - kernel.dimension + 1; depth += stride) {
		
			int rowIndex = 0;
			for (int row = 0; row < matrices[0].rows - kernel.matrices[0].rows + 1; row += stride) {
				
				int colIndex = 0;
				for (int col = 0; col < matrices[0].cols - kernel.matrices[0].cols + 1; col += stride) {
				
					final Tensor3 subtensor = subtensor(row, col, depth, kernel.matrices[0].rows, kernel.matrices[0].cols, dimension);
					final double tensorDot = modifier.apply(subtensor.tensorDot(kernel));
					
					out[depthIndex][rowIndex][colIndex] = tensorDot;
					colIndex++;
				}			
				
				colIndex = 0;
				rowIndex++;
			}
			
			rowIndex = 0;
			depthIndex++;
		}
		
		final Matrix[] matricesOut = new Matrix[out.length];
		for (int i = 0; i < matricesOut.length; i++) {
			matricesOut[i] = new Matrix(out[i]);
		}
		
		return new Tensor3(matricesOut);
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
		final StringBuilder out = new StringBuilder("[");
		
		for (int layer = 0; layer < dimension; layer++) {
			out.append(matrices[layer].toString());
			
			if (layer != dimension - 1) {
				out.append("\n\n");
			} else {
				out.append("]");
			}
		}
		
		return out.toString();
	}
}
