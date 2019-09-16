package main.math.utility;

import java.io.Serializable;

/**
 * Represents a function that operates on two doubles and returns a double result.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface DoubleOperator extends Serializable {
	
	/**
	 * Operates on <code>a</code> and <code>b</code>.
	 * 
	 * @param a first operand
	 * @param b second operand
	 * @return a double
	 */
	double operate(final double a, final double b);
}
