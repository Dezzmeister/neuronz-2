package main.math.test;

import main.math.constructs.Matrix;
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
	private static final float EPSILON = 0.0001f;
	
	/**
	 * Runs a series of tests on the {@link Matrix} class.
	 * 
	 * @param args unused
	 */
	public static final void main(final String[] args) {
		final Matrix m0 = new Matrix(new float[][] {{3, 8}, {4, 6}});
		System.out.println(m0);
		System.out.println("Determinant should be -14");
		System.out.println((m0.determinant() == -14) ? "Test passed!" : "Test failed!");
		System.out.println();
		
		final Matrix m1 = new Matrix(new float[][] {
				 {6, 1, 1},
				 {4, -2, 5},
				 {2, 8, 7}
			});
		System.out.println(m1);
		System.out.println("Determinant should be -306");
		System.out.println((m1.determinant() == -306) ? "Test passed!" : "Test failed!");
		System.out.println();
		
		final Matrix m2 = new Matrix(new float[][] {
			{1, -2.3f, 3, 4},
			{-5, 6, -7.2f, 8},
			{9.49f, 10.06f, 11, 12},
			{13, 14, -15, 16}
		});
		System.out.println("Determinant should be -22423.312");
		System.out.println(TestUtils.closeEnough(m2.determinant(), -22423.312f, EPSILON) ? "Test passed!" : "Test failed!");
	}
}
