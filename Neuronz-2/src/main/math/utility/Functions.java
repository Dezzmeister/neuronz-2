package main.math.utility;

/**
 * Contains important mathematical functions.
 *
 * @author Joe Desmond
 */
public final class Functions {
	
	/**
	 * The sigmoid function.
	 * 
	 * @param x input
	 * @return 1/(1 + e^-x)
	 */
	public static final double sigmoid(double x) {
		return 1.0f/(double)(1.0f + Math.exp(-x));
	}
}
