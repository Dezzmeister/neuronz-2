package main.math.utility;

/**
 * Represents a function that operates on two floats and returns a float result.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface FloatOperator {
	
	/**
	 * Addition, represented as a FloatOperator
	 */
	public static final FloatOperator ADD = (a, b) -> a + b;
	
	/**
	 * Subtraction, represented as a FloatOperator
	 */
	public static final FloatOperator SUBTRACT = (a, b) -> a - b;
	
	/**
	 * Operates on <code>a</code> and <code>b</code>.
	 * 
	 * @param a first operand
	 * @param b second operand
	 * @return a float
	 */
	float operate(final float a, final float b);
}
