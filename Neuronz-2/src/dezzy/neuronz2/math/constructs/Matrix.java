package dezzy.neuronz2.math.constructs;

import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.math.constructs.shape.MatrixShape;
import dezzy.neuronz2.math.utility.DimensionMismatchException;
import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;
import dezzy.neuronz2.math.utility.IndexedGenerator;


/**
 * A matrix with any number of rows and columns.
 *
 * @author Joe Desmond
 */
public final class Matrix extends ElementContainer<Matrix> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -874139546253134819L;

	/**
	 * The values of the matrix
	 */
	private final double[][] values;
	
	/**
	 * Number of rows in the matrix
	 */
	public final int rows;
	
	/**
	 * Number of columns in the matrix. If the matrix is irregular, this will be -1.
	 */
	public final int cols;
	
	/**
	 * The determinant of this Matrix; the placeholder {@link Float#MAX_VALUE} signifies that the determinant has not been calculated or does not exist
	 */
	private double determinant = Float.MAX_VALUE;
	
	/**
	 * Creates a Matrix from the given values. If the rows are not the same length, {@link Matrix#cols} will be -1.
	 * 
	 * @param _values matrix values
	 */
	public Matrix(final double[][] _values) {
		values = _values;
		rows = values.length;
		
		int tempCols = values[0].length;
		
		int firstRowLength = values[0].length;
		for (int row = 1; row < values.length; row++) {
			if (values[row].length != firstRowLength) {
				tempCols = -1;
				break;
			}
		}
		
		cols = tempCols;
	}
	
	/**
	 * Returns the shape of this matrix (number of rows and columns).
	 * 
	 * @return shape of this matrix
	 */
	public MatrixShape shape() {
		return new MatrixShape(rows, cols);
	}
	
	/**
	 * Creates a Matrix from the given row vectors. If the rows are not the same length, {@link Matrix#cols} will be -1.
	 * 
	 * @param vectors rows of the matrix
	 */
	public Matrix(final Vector ... vectors) {
		values = new double[vectors.length][];
		rows = vectors.length;
		
		int tempCols = vectors[0].dimension;
		
		int firstRowLength = vectors[0].dimension;
		for (int row = 0; row < vectors.length; row++) {
			final Vector vector = vectors[row];
			
			if (vector.dimension != firstRowLength) {
				tempCols = -1;
			}
			
			values[row] = new double[vector.dimension];
			
			for (int col = 0; col < vector.dimension; col++) {
				values[row][col] = vector.get(col);
			}
		}
		
		cols = tempCols;
	}
	
	/**
	 * Generates a matrix with the specified size using the given generator function. Calls
	 * {@link IndexedGenerator#generate(int...) generator.generate(row, col)} with every element's row, and column
	 * to obtain the values of the matrix.
	 * 
	 * @param generator generator function, used to generate components of the matrix
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @return a new matrix
	 */
	public static final Matrix generate(final IndexedGenerator generator, final int rows, final int cols) {
		final double[][] out = new double[rows][cols];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				out[row][col] = generator.generate(row, col);
			}
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Returns the element at the given row and column. Does not check to ensure that <code>row</code> and <code>col</code> are
	 * within an acceptable range.
	 * 
	 * @param row must be greater than or equal to 0 and less than {@link Matrix#rows}
	 * @param col must be greater than or equal to 0 and less than {@link Matrix#getRowDimension(row)}
	 * @return the element at the given row and column
	 */
	public final double get(int row, int col) {
		return values[row][col];
	}
	
	/**
	 * Returns a single row vector. Does not check to ensure that <code>row</code> is within an acceptable range.
	 * 
	 * @param row must be greater than or equal to 0 and less than {@link Matrix#rows}
	 * @return the vector at the given row
	 */
	public final Vector getRowVector(int row) {
		return new Vector(values[row]);
	}
	
	/**
	 * Returns a single column vector. Does not check to ensure that <code>col</code> is within an acceptable range or that this
	 * Matrix is regular.
	 * 
	 * @param col must be greater than or equal to 0 and less that {@link Matrix#cols}
	 * @return the vector at the given column
	 */
	public final Vector getColVector(int col) {
		final double[] result = new double[values.length];
		
		for (int row = 0; row < values.length; row++) {
			result[row] = values[row][col];
		}
		
		return new Vector(result);
	}
	
	/**
	 * Copies the entire matrix row-wise into the given array.
	 * 
	 * @param array array to copy this matrix to
	 * @param destIndex starting index in <code>array</code> to copy elements to
	 */
	public final void copyTo(final double[] array, final int destIndex) {
		for (int row = 0; row < rows; row++) {
			System.arraycopy(values[row], 0, array, destIndex + (row * cols), cols);
		}
	}
	
	/**
	 * Multiplies this Matrix with another. Produces a matrix with the same number of rows as this matrix and {@link Matrix#rows other.cols} columns.
	 * 
	 * @param other other matrix to be multiplied
	 * @return the product of <code>this</code> and <code>other</code>
	 */
	public final Matrix multiply(final Matrix other) {
		final double[][] values = new double[rows][other.cols];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < other.cols; col++) {
				final Vector rowVec = getRowVector(row);
				final Vector colVec = other.getColVector(col);
				
				values[row][col] = rowVec.innerProduct(colVec);
			}
		}
		
		return new Matrix(values);
	}
	
	/**
	 * Convolves this matrix with a kernel and applies a modifier function to each resulting element. This is technically
	 * the cross-correlation operation because the kernel is never flipped.
	 * 
	 * @param kernel kernel matrix
	 * @param stride kernel stride
	 * @param modifier modifier function
	 * @return convolution of this matrix and the kernel, with modifier function applied
	 */
	public final Matrix convolve(final Matrix kernel, final int stride, final DoubleApplier modifier) {
		final double[][] out = new double[((rows - kernel.rows) / stride) + 1][((cols - kernel.cols) / stride) + 1];
		
		int rowIndex = 0;
		for (int row = 0; row < rows - kernel.rows + 1; row += stride) {
			
			int colIndex = 0;
			for (int col = 0; col < cols - kernel.cols + 1; col += stride) {
				final Matrix submatrix = submatrix(row, col, kernel.rows, kernel.cols);
				final double frobeniusProduct = modifier.apply(submatrix.frobenius(kernel));
				
				out[rowIndex][colIndex] = frobeniusProduct;
				colIndex++;
			}
			colIndex = 0;
			rowIndex++;
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Applies a pooling transformation to this matrix, and returns a smaller matrix with the result. 
	 * If there are not enough elements in the window (i.e., the window overlaps
	 * the border of the matrix), the pooling operation will not be applied to the extra elements.
	 * 
	 * @param windowRows number of rows in the pooling window
	 * @param windowCols number of columns in the pooling window
	 * @param operation pooling function to apply to the window
	 * @return a matrix with the pooling function applied to every consecutive window over the image
	 */
	public final Matrix poolingTransform(final int windowRows, final int windowCols, final int rowStride, final int colStride, final PoolingOperation operation) {
		final double[][] out = new double[((rows - windowRows) / rowStride) + 1][((cols - windowCols) / colStride) + 1];
		
		int rowIndex = 0;
		for (int row = 0; row < rows - windowRows + 1; row += rowStride) {
			
			int colIndex = 0;
			for (int col = 0; col < cols - windowCols + 1; col += colStride) {
				final Matrix submatrix = submatrix(row, col, windowRows, windowCols);
				final double result = operation.condense(submatrix);
				
				out[rowIndex][colIndex] = result;				
				colIndex++;
			}
			colIndex = 0;
			rowIndex++;
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Expands this matrix to be the same size as the given matrix, then multiplies them element-wise.
	 * Used in backpropagation for {@linkplain dezzy.neuronz2.cnn.layers.PoolingLayer pooling layers}.
	 * In pooling layers, this function is called by the smaller matrix, which is the derivative of the
	 * error with respect to the pooling layer's output. The argument to this function is the 
	 * {@linkplain PoolingResult#modifiedInput modified input} (returned by 
	 * {@link #poolingTransform(int, int, int, int, PoolingOperation) poolingTransform()}).
	 * 
	 * @param other larger matrix
	 * @return a matrix with the same size as the larger matrix
	 */
	public final Matrix expandAndMultiply(final Matrix other) {
		final double[][] out = new double[other.rows][other.cols];
		
		final int rowStride = other.rows / rows;
		final int colStride = other.cols / cols;
		
		int smallRowIndex = 0;
		int smallColIndex = 0;
		
		// These counts are used to avoid the modulus operation and floating point addition
		int rowCount = 0;
		int colCount = 0;
		
		for (int row = 0; row < other.rows; row++) {
			for (int col = 0; col < other.cols; col++) {
				out[row][col] = other.values[row][col] * values[smallRowIndex][smallColIndex];
				
				colCount++;
				if (colCount == colStride) {
					smallColIndex++;
					colCount = 0;
				}
			}
			
			rowCount++;
			if (rowCount == rowStride) {
				smallRowIndex++;
				rowCount = 0;
			}
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Pads this matrix with a constant value, giving a larger matrix with the padding. The new matrix has a size of
	 * ({@link #rows} + <code>(padRows * 2)</code>) by ({@link #cols} + <code>(padCols * 2</code>). A copy of the values
	 * in the current matrix is centered in the new matrix.
	 * 
	 * @param padRows number of rows to pad with (padding at top and bottom of matrix)
	 * @param padCols number of columns to pad with (padding at left and right of matrix)
	 * @param value value of each padded location
	 * @return a version of this matrix with padding added
	 */
	public final Matrix pad(final int padRows, final int padCols, final double value) {
		final double[][] out = new double[rows + (2 * padRows)][cols + (2 * padCols)];
		
		for (int row = 0; row < padRows; row++) {
			for (int col = 0; col < out[0].length; col++) {
				out[row][col] = value;
			}
		}
		
		for (int row = rows + padRows; row < out.length; row++) {
			for (int col = 0; col < out[0].length; col++) {
				out[row][col] = value;
			}
		}
		
		for (int col = 0; col < padCols; col++) {
			for (int row = padRows; row < rows + padRows; row++) {
				out[row][col] = value;
			}
		}
		
		for (int col = cols + padCols; col < out[0].length; col++) {
			for (int row = padRows; row < rows + padRows; row++) {
				out[row][col] = value;
			}
		}
		
		for (int row = padRows; row < rows + padRows; row++) {
			System.arraycopy(values[row - padRows], 0, out[row], padCols, values[0].length);
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Pads this matrix with zeroes. Functionally equivalent to calling 
	 * {@link #pad(int, double) pad(padRows, padCols, 0.0)}. The difference between the two
	 * is that this exploits the fact that arrays are initially zeroed, making it faster than a call to
	 * {@link #pad(int, double) pad(padRows, padCols, 0.0)}.
	 * 
	 * @param padRows number of rows to pad with (padding at top and bottom of matrix)
	 * @param padCols number of columns to pad with (padding at left and right of matrix)
	 * @return a version of this matrix with padding added (zeroes)
	 */
	public final Matrix padZero(final int padRows, final int padCols) {
		final double[][] out = new double[rows + (2 * padRows)][cols + (2 * padCols)];
		
		for (int row = padRows; row < rows + padRows; row++) {
			System.arraycopy(values[row - padRows], 0, out[row], padCols, values[0].length);
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Flips a matrix horizontally: mirrors elements over a horizontal line splitting the matrix
	 * into two equal parts.
	 * 
	 * @return a version of this matrix flipped horizontally
	 */
	public final Matrix flipHorizontal() {
		final double[][] out = new double[rows][cols];
		
		for (int row = 0; row < rows; row++) {
			System.arraycopy(values[row], 0, out[rows - row - 1], 0, cols);
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Flips a matrix vertically: mirrors elements over a vertical line splitting the matrix
	 * into two equal parts.
	 * 
	 * @return a version of this matrix flipped vertically
	 */
	public final Matrix flipVertical() {
		final double[][] out = new double[rows][cols];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				out[row][col] = values[row][cols - col - 1];
			}
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Rotates this matrix by 180 degrees. Functionally equivalent to calling {@link #flipHorizontal()} 
	 * followed by {@link #flipVertical()}, or vice versa. The difference is that this function is faster because
	 * it requires half the memory allocations.
	 * 
	 * @return a version of this matrix rotates by 180 degrees
	 */
	public final Matrix rotate180() {
		final double[][] out = new double[rows][cols];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				out[row][col] = values[rows - row - 1][cols - col - 1];
			}
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Returns the sum of every element in this matrix.
	 * 
	 * @return sum of every element in this matrix
	 */
	public final double sum() {
		double sum = 0;
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				sum += values[row][col];
			}
		}
		
		return sum;
	}
	
	/**
	 * Gets a submatrix within this matrix.
	 * 
	 * @param row starting row (inclusive) within this matrix
	 * @param col starting column (inclusive) within this matrix
	 * @param subRows number of rows in the submatrix
	 * @param subCols number of columns in the submatrix
	 * @return submatrix with size <code>[subRows][subCols]</code>
	 */
	public final Matrix submatrix(final int row, final int col, final int subRows, final int subCols) {
		final double[][] out = new double[subRows][subCols];
		
		for (int rowIndex = 0; rowIndex < subRows; rowIndex++) {
			for (int colIndex = 0; colIndex < subCols; colIndex++) {
				out[rowIndex][colIndex] = values[row + rowIndex][col + colIndex];
			}
		}
		
		return new Matrix(out);
	}
	
	/**
	 * Computes the Frobenius inner product of two matrices. This works exactly like the dot
	 * product for vectors, but is extended to matrices.
	 * 
	 * @param other other matrix with the same dimensions as this one
	 * @return scalar Frobenius inner product
	 */
	public final double frobenius(final Matrix other) {
		if (rows != other.rows || cols != other.cols) {
			throw new DimensionMismatchException("Matrices must be the same size!");
		}
		
		double product = 0;
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				product += values[row][col] * other.values[row][col];
			}
		}
		
		return product;
	}
	
	/**
	 * Returns the transpose of this Matrix (every value is mirrored over the diagonal).
	 * 
	 * @return the transpose of this Matrix
	 */
	public final Matrix transpose() {
		final double[][] result = new double[cols][rows];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[col][row] = values[row][col];
			}
		}
		
		return new Matrix(result);
	}
	
	/**
	 * Multiplies this Matrix by a column Vector.
	 * 
	 * @param vector column vector
	 * @return <code>this * vector</code>
	 */
	public final Vector multiply(final Vector vector) {
		if (vector.dimension != cols) {
			throw new DimensionMismatchException("Vector must have the same number of components as the matrix's number of columns!");
		}
		
		final double[] result = new double[rows];
		
		for (int row = 0; row < rows; row++) {
			result[row] = vector.innerProduct(getRowVector(row));
		}
		
		return new Vector(result);
	}
	
	/**
	 * Returns the dimension of a single row vector. Does not check to ensure that <code>row</code> is within an acceptable range.
	 * 
	 * @param row must be greater than or equal to 0 and less than {@link Matrix#rows}
	 * @return the dimension of the row vector at the given row
	 */
	public final double getRowDimension(int row) {
		return values[row].length;
	}
	
	/**
	 * Recursively calculates the determinant of a Matrix if it has not yet been calculated for this Matrix. Otherwise returns the previously calculated determinant.
	 * Throws a {@link DimensionMismatchException} if this Matrix is not a square matrix.
	 * 
	 * @return the determinant of this Matrix
	 */
	public final double determinant() {		
		if (!isSquare()) {
			throw new DimensionMismatchException("Determinant can only be calculated for a square matrix!");
		}
		
		//Return the determinant if it has already been calculated, since the values of this Matrix are immutable
		if (determinant != Float.MAX_VALUE) {
			return determinant;
		}
		
		if (rows == 2) {
			determinant = values[0][0] * values[1][1] - values[1][0] * values[0][1];
			return determinant;
		} else if (rows <= 1) {
			return 0;
		} else {
			double determinant = 0;
			
			for (int col = 0; col < cols; col++) {
				final double[][] smallMatrixValues = new double[rows - 1][cols - 1];
				
				for (int row = 1; row < rows; row++) {
					
					int colIndex = 0;
					for (int col2 = 0; col2 < cols; col2++) {
						if (col != col2) {
							smallMatrixValues[row - 1][colIndex] = values[row][col2];
							colIndex++;
						}
					}
				}
				
				double smallMatrixDeterminant = new Matrix(smallMatrixValues).determinant();
				double colElement = values[0][col];
				double elementDeterminant = smallMatrixDeterminant * colElement;
				
				determinant += (col % 2 == 0) ? elementDeterminant : -elementDeterminant;
			}
			
			return determinant;
		}
	}
	
	/**
	 * Returns true if this Matrix is a square matrix.
	 * 
	 * @return true if this Matrix has an equal number of rows and columns.
	 */
	public boolean isSquare() {
		return rows == cols;
	}
	
	/**
	 * Applies an operation to each element of this Matrix and another. Throws a {@link DimensionMismatchException} if the matrices do not have the same dimensions.
	 * Elements from this Matrix are passed in as <code>a</code> in <code>operator</code>, and elements from the other Matrix are passed in as <code>b</code>.
	 * 
	 * @param other other Matrix
	 * @param operator operation to be performed on each element
	 * @return a new Matrix with the result of the operation applied to each element and the same dimensions as this Matrix
	 */
	@Override
	public final Matrix elementOperation(final Matrix other, final DoubleOperator operator) {
		if (!isSameDimensionsAs(other)) {
			throw new DimensionMismatchException("Matrices must have the same dimensions to perform element operations!");
		}
		
		final double[][] result = new double[rows][cols];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = operator.operate(values[row][col], other.values[row][col]);
			}
		}
		
		return new Matrix(result);
	}
	
	@Override
	public final Matrix transform(final DoubleApplier operator) {
		final double[][] result = new double[rows][cols];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = operator.apply(values[row][col]);
			}
		}
		
		return new Matrix(result);
	}
	
	/**
	 * An index into this matrix.
	 *
	 * @author Joe Desmond
	 */
	public final class Index {
		
		/**
		 * The value at this index
		 */
		public final double value;
		
		/**
		 * The row of <code>value</code>
		 */
		public final int row;
		
		/**
		 * The column of <code>value</code>
		 */
		public final int col;
		
		/**
		 * Constructs an index into this matrix, pointing at the given value located
		 * at the given row and column.
		 * 
		 * @param _value value
		 * @param _row row
		 * @param _col column
		 */
		public Index(final double _value, final int _row, final int _col) {
			value = _value;
			row = _row;
			col = _col;
		}
	}
	
	/**
	 * Returns the location and value of the maximum element in this
	 * matrix.
	 * 
	 * @return index to the maximum element in this matrix
	 */
	public final Index max() {
		double max = Double.NEGATIVE_INFINITY;
		int maxRow = 0;
		int maxCol = 0;
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				final double value = values[row][col];
				
				if (value > max) {
					max = value;
					maxRow = row;
					maxCol = col;
				}
			}
		}
		
		return new Index(max, maxRow, maxCol);
	}
	
	/**
	 * Returns the location and value of the minimum element in this
	 * matrix.
	 * 
	 * @return index to the minimum element in this matrix
	 */
	public final Index min() {
		double min = Double.POSITIVE_INFINITY;
		int minRow = 0;
		int minCol = 0;
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				final double value = values[row][col];
				
				if (value < min) {
					min = value;
					minRow = row;
					minCol = col;
				}
			}
		}
		
		return new Index(min, minRow, minCol);
	}
	
	/**
	 * Returns true if this matrix and another have the same number of rows and the same number of columns.
	 * 
	 * @param other other matrix
	 * @return true if this matrix and the other have the same dimensions
	 */
	public final boolean isSameDimensionsAs(final Matrix other) {
		return rows == other.rows && cols == other.cols && cols != -1 && other.cols != -1;
	}
	
	/**
	 * Outputs this Matrix into an easily readable format.
	 */
	@Override
	public String toString() {
		String out = "";
		
		for (int row = 0; row < rows; row++) {
			out += "[";
			
			for (int col = 0; col < cols; col++) {
				out += values[row][col];
				
				if (col != cols - 1) {
					out += " ";
				}
			}
			
			out += "]";
			
			if (row != rows - 1) {
				out += "\n";
			}
		}
		
		return out;
	}
}
