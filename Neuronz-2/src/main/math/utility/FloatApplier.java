package main.math.utility;

/**
 * Represents a function that takes one double and returns a double result.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface FloatApplier {
	
	/**
	 * Applies an operation to a double.
	 * 
	 * @param a input
	 * @return result
	 */
	double apply(final double a);
}
