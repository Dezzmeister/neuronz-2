package test;

/**
 * Contains general utility functions for running tests.
 *
 * @author Joe Desmond
 */
public final class TestUtils {
	
	/**
	 * Returns true if the test value is close enough to the target value to be considered equal.
	 * 
	 * @param test test result
	 * @param target expected result
	 * @param epsilon acceptable margin of error
	 * @return true if the absolute value of <code>(test - target)</code> is less than or equal to <code>epsilon<code>
	 */
	public static final boolean closeEnough(final double test, final double target, final double epsilon) {
		return Math.abs(target - test) <= epsilon;
	}
}
