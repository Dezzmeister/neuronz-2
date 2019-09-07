package main.math.utility;

/**
 * Represents a function that operates on two floats and returns a float result.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface FloatOperator {
	
	/**
	 * Operates on <code>a</code> and <code>b</code>.
	 * 
	 * @param a first operand
	 * @param b second operand
	 * @return a float
	 */
	float operate(final float a, final float b);
}
