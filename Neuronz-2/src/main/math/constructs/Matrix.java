package main.math.constructs;

import main.math.utility.DimensionMismatchException;
import main.math.utility.FloatApplier;
import main.math.utility.FloatOperator;

/**
 * A matrix with any number of rows and columns.
 *
 * @author Joe Desmond
 */
public final class Matrix extends ElementContainer<Matrix> {
	/**
	 * The values of the matrix
	 */
	private final float[][] values;
	
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
	private float determinant = Float.MAX_VALUE;
	
	/**
	 * Creates a Matrix from the given values. If the rows are not the same length, {@link Matrix#cols} will be -1.
	 * 
	 * @param _values matrix values
	 */
	public Matrix(final float[][] _values) {
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
	 * Creates a Matrix from the given row vectors. If the rows are not the same length, {@link Matrix#cols} will be -1.
	 * 
	 * @param vectors rows of the matrix
	 */
	public Matrix(final Vector ... vectors) {
		values = new float[vectors.length][];
		rows = vectors.length;
		
		int tempCols = vectors[0].dimension;
		
		int firstRowLength = vectors[0].dimension;
		for (int row = 0; row < vectors.length; row++) {
			final Vector vector = vectors[row];
			
			if (vector.dimension != firstRowLength) {
				tempCols = -1;
			}
			
			values[row] = new float[vector.dimension];
			
			for (int col = 0; col < vector.dimension; col++) {
				values[row][col] = vector.get(col);
			}
		}
		
		cols = tempCols;
	}
	
	/**
	 * Returns the element at the given row and column. Does not check to ensure that <code>row</code> and <code>col</code> are
	 * within an acceptable range.
	 * 
	 * @param row must be greater than or equal to 0 and less than {@link Matrix#rows}
	 * @param col must be greater than or equal to 0 and less than {@link Matrix#getRowDimension(row)}
	 * @return the element at the given row and column
	 */
	public final float get(int row, int col) {
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
		final float[] result = new float[values.length];
		
		for (int row = 0; row < values.length; row++) {
			result[row] = values[row][col];
		}
		
		return new Vector(result);
	}
	
	/**
	 * Multiplies this Matrix with another. Produces a matrix with the same number of rows as this matrix and {@link Matrix#rows other.cols} columns.
	 * 
	 * @param other other matrix to be multiplied
	 * @return the product of <code>this</code> and <code>other</code>
	 */
	public final Matrix multiply(final Matrix other) {
		final float[][] values = new float[rows][other.cols];
		
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
	 * Multiplies this Matrix by a column Vector.
	 * 
	 * @param vector column vector
	 * @return <code>this * vector</code>
	 */
	public final Vector multiply(final Vector vector) {
		if (vector.dimension != cols) {
			throw new DimensionMismatchException("Vector must have the same number of components as the matrix's number of columns!");
		}
		
		final float[] result = new float[rows];
		
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
	public final float getRowDimension(int row) {
		return values[row].length;
	}
	
	/**
	 * Recursively calculates the determinant of a Matrix if it has not yet been calculated for this Matrix. Otherwise returns the previously calculated determinant.
	 * Throws a {@link DimensionMismatchException} if this Matrix is not a square matrix.
	 * 
	 * @return the determinant of this Matrix
	 */
	public final float determinant() {		
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
			float determinant = 0;
			
			for (int col = 0; col < cols; col++) {
				final float[][] smallMatrixValues = new float[rows - 1][cols - 1];
				
				for (int row = 1; row < rows; row++) {
					
					int colIndex = 0;
					for (int col2 = 0; col2 < cols; col2++) {
						if (col != col2) {
							smallMatrixValues[row - 1][colIndex] = values[row][col2];
							colIndex++;
						}
					}
				}
				
				float smallMatrixDeterminant = new Matrix(smallMatrixValues).determinant();
				float colElement = values[0][col];
				float elementDeterminant = smallMatrixDeterminant * colElement;
				
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
	public final Matrix elementOperation(final Matrix other, final FloatOperator operator) {
		if (!isSameDimensionsAs(other)) {
			throw new DimensionMismatchException("Matrices must have the same dimensions to perform element operations!");
		}
		
		final float[][] result = new float[rows][cols];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = operator.operate(values[row][col], other.values[row][col]);
			}
		}
		
		return new Matrix(result);
	}
	
	@Override
	public final Matrix transform(final FloatApplier operator) {
		final float[][] result = new float[rows][cols];
		
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = operator.apply(values[row][col]);
			}
		}
		
		return new Matrix(result);
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
