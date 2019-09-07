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
	public static final float sigmoid(float x) {
		return 1.0f/(float)(1.0f + Math.exp(-x));
	}
}
