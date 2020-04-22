package dezzy.neuronz2.cnn.pooling;

import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Matrix.Index;

/**
 * The max pooling operation.
 *
 * @author Joe Desmond
 */
public final class MaxPooling implements PoolingOperation {

	@Override
	public final double condense(final Matrix matrix) {
		double max = Double.NEGATIVE_INFINITY;
		
		for (int row = 0; row < matrix.rows; row++) {
			for (int col = 0; col < matrix.cols; col++) {
				final double value = matrix.get(row, col);
				
				if (value > max) {
					max = value;
				}
			}
		}
		
		return max;
	}
	
	@Override
	public final Matrix backprop(final Matrix latestInput, final Matrix derivative, final int rowStride, final int colStride) {
		final double[][] out = new double[latestInput.rows][latestInput.cols];
		
		int rowIndex = 0;
		int colIndex = 0;
		
		for (int smallRow = 0; smallRow < derivative.rows; smallRow++) {
			for (int smallCol = 0; smallCol < derivative.cols; smallCol++) {
				final Matrix window = latestInput.submatrix(rowIndex, colIndex, derivative.rows, derivative.cols);
				
				final Index max = window.max();
				out[max.row][max.col] += derivative.get(smallRow, smallCol);
				
				colIndex += colStride;
			}
			
			rowIndex += rowStride;
			colIndex = 0;
		}
		
		return new Matrix(out);
	}	
}
