package main.math.test;

import main.math.constructs.Matrix;

/**
 * Testing hardware speed by multiplying large matrices
 *
 * @author Joe Desmond
 */
public final class MatrixBenchmark {
	
	public static final void main(final String[] args) {
		final double[][] doubles0 = new double[500][500];
		final double[][] doubles1 = new double[500][500];
		
		fillRandom(doubles0);
		fillRandom(doubles1);
		
		final Matrix mat0 = new Matrix(doubles0);
		final Matrix mat1 = new Matrix(doubles1);
		
		long startMillis = System.currentTimeMillis();
		mat0.multiply(mat1);
		long endMillis = System.currentTimeMillis();
		System.out.println((endMillis - startMillis) + " ms");
	}
	
	/**
	 * Fill a 2D array with random values from -1 to 1, assume a uniform array
	 * 
	 * @param doubles double array
	 */
	private static final void fillRandom(double[][] doubles) {
		for (int row = 0; row < doubles.length; row++) {
			for (int col = 0; col < doubles[0].length; col++) {
				doubles[row][col] = (Math.random() * 2) - 1;
			}
		}
	}
}
