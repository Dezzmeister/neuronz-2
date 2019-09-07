package main.math.utility;

/**
 * Represents a function that takes one float and returns a float result.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface FloatApplier {
	
	/**
	 * Applies an operation to a float.
	 * 
	 * @param a input
	 * @return result
	 */
	float apply(final float a);
}
