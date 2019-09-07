package main.math.constructs;

import main.math.utility.FloatApplier;
import main.math.utility.FloatOperator;

/**
 * Represents anything that contains a collection of float elements.
 *
 * @author Joe Desmond
 */
public abstract class ElementContainer<T> {
	
	/**
	 * Performs an operation with each element of <code>this</code> and <code>other</code>.
	 * 
	 * @param other something containing float elements
	 * @param operator operation to be performed on each pair of elements
	 * @return the application of <code>operator</code> on <code>this</code> and <code>other</code>
	 */
	public abstract T elementOperation(final T other, final FloatOperator operator);
	
	/**
	 * Performs an operation on each element of <code>this</code>.
	 * 
	 * @param transformation operation to be performed on each element
	 * @return the application of <code>operator</code> on <code>this</code>
	 */
	public abstract T transform(final FloatApplier transformation);
	
	/**
	 * Addition, represented as a FloatOperator
	 */
	private static final FloatOperator ADD = (a, b) -> a + b;
	
	/**
	 * Subtraction, represented as a FloatOperator
	 */
	private static final FloatOperator SUBTRACT = (a, b) -> a - b;
	
	private static final FloatOperator MULTIPLY = (a, b) -> a * b;
	
	private static final FloatOperator DIVIDE = (a, b) -> a / b;
	
	/**
	 * Adds <code>this</code> to <code>other</code>, element-wise.
	 * 
	 * @param other to be added to <code>this</code>
	 * @return <code>(this + other)</code> for each element
	 */
	public final T plus(final T other) {
		return elementOperation(other, ADD);
	}
	
	/**
	 * Subtracts <code>other</code> from <code>this</code>, element-wise.
	 * 
	 * @param other to be subtracted from <code>this</code>
	 * @return <code>(this - other)</code> for each element
	 */
	public final T minus(final T other) {
		return elementOperation(other, SUBTRACT);
	}
	
	/**
	 * Computes the hadamard product (element-wise multiplication) of <code>this</code> and <code>other</code>.
	 * 
	 * @param other to be multiplied to <code>this</code>, element-wise
	 * @return <code>(this * other)</code> for each element
	 */
	public final T hadamard(final T other) {
		return elementOperation(other, MULTIPLY);
	}
	
	/**
	 * Divides <code>this</code> by <code>other</code>, element-wise.
	 * 
	 * @param other to be divided by
	 * @return <code>(this / other)</code> for each element
	 */
	public final T elementDivide(final T other) {
		return elementOperation(other, DIVIDE);
	}
}
