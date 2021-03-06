package dezzy.neuronz2.math.test;

import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;
import test.TestUtils;

/**
 * A series of tests for methods in {@link Matrix}.
 *
 * @author Joe Desmond
 */
public final class MatrixTest {
	/**
	 * Acceptable margin of error when evaluating tests.
	 */
	private static final double EPSILON = 0.0001f;
	
	/**
	 * Runs a series of tests on the {@link Matrix} class.
	 * 
	 * @param args unused
	 */
	public static final void main(final String[] args) {
		final Matrix m0 = new Matrix(new double[][] {{3, 8}, {4, 6}});
		System.out.println(m0);
		System.out.println("Determinant should be -14");
		System.out.println((m0.determinant() == -14) ? "Test passed!" : "Test failed!");
		System.out.println();
		
		final Matrix m1 = new Matrix(new double[][] {
				 {6, 1, 1},
				 {4, -2, 5},
				 {2, 8, 7}
			});
		System.out.println(m1);
		System.out.println("Determinant should be -306");
		System.out.println((m1.determinant() == -306) ? "Test passed!" : "Test failed!");
		System.out.println();
		
		final Matrix m2 = new Matrix(new double[][] {
			{1, -2.3f, 3, 4},
			{-5, 6, -7.2f, 8},
			{9.49f, 10.06f, 11, 12},
			{13, 14, -15, 16}
		});
		System.out.println("Determinant should be -22423.312");
		System.out.println(TestUtils.closeEnough(m2.determinant(), -22423.312f, EPSILON) ? "Test passed!" : "Test failed!");
		System.out.println();
		
		System.out.println("M3: ");
		final Matrix m3 = new Matrix(new double[][] {
			{1, 2, 3},
			{4, 5, 6}
		});
		System.out.println(m3);
		
		System.out.println("\nM4: ");
		final Matrix m4 = new Matrix(new double[][] {
			{7, 8},
			{9, 10},
			{11, 12}
		});
		System.out.println(m4);
		
		System.out.println("\nM3 * M4: ");
		System.out.println(m3.multiply(m4));
		System.out.println();
		
		System.out.println("M5: ");
		final Matrix m5 = new Matrix(new double[][] {
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
			{10, 11, 12}
		});
		System.out.println(m5);
		System.out.println();
		
		System.out.println("V0: ");
		final Vector v0 = new Vector(-2, 1, 0);
		System.out.println(v0);
		System.out.println();
		
		System.out.println("M5 * v0: " + m5.multiply(v0));
		
		System.out.println("\nConvolution:");
		final Matrix kernel = new Matrix(new double[][] {
			{-1., 0.5},
			{0.2, 3.}
		});
		
		final Matrix image = new Matrix(new double[][] {
			{4, 3, 2, 1},
			{1, 2, 0, 4},
			{6, 1, 3, 2},
			{7, 8, 9, 9}
		});
		
		System.out.println(image.convolve(kernel, 1, d -> d));
		
		System.out.println("\nMax Pooling:");
		final Matrix maxPooled = image.poolingTransform(2, 2, 2, 2, PoolingOperation.MAX_POOLING);
		System.out.println(maxPooled);
		
		System.out.println("\nMax Pooling Expansion:");
		System.out.println(PoolingOperation.MAX_POOLING.backprop(image, maxPooled, 2, 2, 2, 2));
		
		System.out.println("\nZero padding, 3 rows and 3 cols:");
		System.out.println(image.padZero(3, 3));
	}
}
