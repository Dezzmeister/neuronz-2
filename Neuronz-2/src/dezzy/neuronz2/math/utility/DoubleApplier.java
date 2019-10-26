package dezzy.neuronz2.math.utility;

import java.io.Serializable;

/**
 * Represents a function that takes one double and returns a double result.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface DoubleApplier extends Serializable {
	
	/**
	 * Applies an operation to a double.
	 * 
	 * @param a input
	 * @return result
	 */
	double apply(final double a);
}
