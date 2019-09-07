package main.math.constructs;

import main.math.utility.DimensionMismatchException;
import main.math.utility.FloatOperator;

/**
 * A matrix with any number of rows and columns.
 *
 * @author Joe Desmond
 */
public final class Matrix {
	/**
	 * The values of the matrix
	 */
	private final float[][] values;
	
	/**
	 * Number of rows in the matrix
	 */
	public final int rows;
	
	/**
	 * Number of columns in the matrix
	 */
	public final int cols;
	
	/**
	 * The determinant of this Matrix, the placeholder {@link Float#MAX_VALUE} signifies that the determinant has not been calculated or does not exist
	 */
	private float determinant = Float.MAX_VALUE;
	
	/**
	 * Creates a Matrix from the given values.
	 * 
	 * @param _values matrix values
	 */
	public Matrix(final float[][] _values) {
		values = _values;
		rows = values.length;
		cols = values[0].length;
	}
	
	/**
	 * Returns the element at the given row and column. Does not check to ensure that <code>row</code> and <code>col</code> are
	 * within an acceptable range.
	 * 
	 * @param row must be greater than or equal to 0 and less than {@link Matrix#rows}
	 * @param col must be greater than or equal to 0 and less than {@link Matrix#cols}
	 * @return the element at the given row and column
	 */
	public float get(int row, int col) {
		return values[row][col];
	}
	
	/**
	 * Calculates the determinant of a Matrix if it has not yet been calculated for this Matrix. Otherwise returns the previously calculated determinant.
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
	 * Addition, represented as a {@link FloatOperator}
	 */
	private static final FloatOperator ADD = (a, b) -> a + b;
	
	/**
	 * Subtraction, represented as a {@link FloatOperator}
	 */
	private static final FloatOperator SUBTRACT = (a, b) -> a - b;
	
	/**
	 * Adds this Matrix to another.
	 * 
	 * @param other Matrix to be added to this Matrix
	 * @return sum of <code>this</code> and <code>other</code>
	 */
	public final Matrix plus(final Matrix other) {
		return elementOperation(other, ADD);
	}
	
	/**
	 * Subtracts another Matrix from this Matrix.
	 * 
	 * @param other Matrix to be subtracted from this Matrix
	 * @return result of <code>(this - other)</code>
	 */
	public final Matrix minus(final Matrix other) {
		return elementOperation(other, SUBTRACT);
	}
	
	/**
	 * Applies an operation to each element of this Matrix and another. Throws a {@link DimensionMismatchException} if the matrices do not have the same dimensions.
	 * Elements from this Matrix are passed in as <code>a</code> in <code>operator</code>, and elements from the other Matrix are passed in as <code>b</code>.
	 * 
	 * @param other other Matrix
	 * @param operator operation to be performed on each element
	 * @return a new Matrix with the result of the operation applied to each element and the same dimensions as this Matrix
	 */
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
	
	/**
	 * Returns true if this matrix and another have the same number of rows and the same number of columns.
	 * 
	 * @param other other matrix
	 * @return true if this matrix and the other have the same dimensions
	 */
	public final boolean isSameDimensionsAs(final Matrix other) {
		return rows == other.rows && cols == other.cols;
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
