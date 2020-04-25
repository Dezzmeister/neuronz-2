package dezzy.neuronz2.math.constructs.shape;

import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * The shape of a matrix, contains the number of rows and columns in the matrix.
 *
 * @author Joe Desmond
 */
public final class MatrixShape extends Shape<Matrix> {
	
	/**
	 * Number of rows in the matrix
	 */
	public final int rows;
	
	/**
	 * Number of columns in the matrix
	 */
	public final int cols;

	/**
	 * Constructs an object containing the shape of a matrix. The matrix itself can be generated
	 * with {@link #generate(IndexedGenerator)}.
	 * 
	 * @param _rows number of rows in the matrix
	 * @param _cols number of columns in the matrix
	 */
	public MatrixShape(final int _rows, final int _cols) {
		rows = _rows;
		cols = _cols;
	}
	
	@Override
	public Matrix generate(final IndexedGenerator generator) {
		return Matrix.generate(generator, rows, cols);
	}
	
}
