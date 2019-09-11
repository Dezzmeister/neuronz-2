package main.math.utility;

/**
 * Represents a function that operates on two doubles and returns a double result.
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
	 * @return a double
	 */
	double operate(final double a, final double b);
}
