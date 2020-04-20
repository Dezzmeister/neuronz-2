package dezzy.neuronz2.cnn.pooling;

import dezzy.neuronz2.math.constructs.Matrix;

/**
 * The max pooling operation.
 *
 * @author Joe Desmond
 */
public final class MaxPooling implements PoolingOperation {

	@Override
	public final SliceResult condense(final Matrix matrix) {
		final double[][] modified = new double[matrix.rows][matrix.cols];
		
		double max = Double.NEGATIVE_INFINITY;
		int maxRow = 0;
		int maxCol = 0;
		
		for (int row = 0; row < matrix.rows; row++) {
			for (int col = 0; col < matrix.cols; col++) {
				final double value = matrix.get(row, col);
				
				if (value > max) {
					max = value;
					maxRow = row;
					maxCol = col;
				}
			}
		}
		
		modified[maxRow][maxCol] = max;
		return new SliceResult(max, new Matrix(modified));
	}
	
}
