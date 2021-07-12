package dezzy.neuronz2.math.constructs;

import java.io.Serializable;

import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;

/**
 * Represents anything that contains a collection of <code>double</code> elements. The general pattern
 * here is to keep subclasses of ElementContainer immutable. See {@link Vector} for an example.
 *
 * @author Joe Desmond
 */
public abstract class ElementContainer<T> implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2655896537829731567L;

	/**
	 * Performs an operation with each element of <code>this</code> and <code>other</code>.
	 * 
	 * @param other something containing double elements
	 * @param operator operation to be performed on each pair of elements
	 * @return the application of <code>operator</code> on <code>this</code> and <code>other</code>
	 */
	public abstract T elementOperation(final T other, final DoubleOperator operator);
	
	/**
	 * Performs an operation on each element of <code>this</code>.
	 * 
	 * @param transformation operation to be performed on each element
	 * @return the application of <code>operator</code> on <code>this</code>
	 */
	public abstract T transform(final DoubleApplier transformation);
	
	/**
	 * Addition, represented as a DoubleOperator
	 */
	private static final DoubleOperator ADD = (a, b) -> a + b;
	
	/**
	 * Subtraction, represented as a DoubleOperator
	 */
	private static final DoubleOperator SUBTRACT = (a, b) -> a - b;
	
	/**
	 * Multiplication, represented as a DoubleOperator
	 */
	private static final DoubleOperator MULTIPLY = (a, b) -> a * b;
	
	/**
	 * Division, represented as a DoubleOperator
	 */
	private static final DoubleOperator DIVIDE = (a, b) -> a / b;
	
	/**
	 * Adds <code>this</code> to <code>other</code>, element-wise.
	 * 
	 * @param other to be added to <code>this</code>
	 * @return <code>(this + other)</code> for each element
	 */
	public T plus(final T other) {
		return elementOperation(other, ADD);
	}
	
	/**
	 * TODO: FIX THIS HACK
	 * Unsafe version of {@link #plus(Object)} that exists for the parallel layer framework.
	 * This should be fixed as soon as possible.
	 * 
	 * @param other other element container
	 * @return this plus the other
	 */
	@SuppressWarnings("unchecked")
	public T unsafePlus(final ElementContainer<?> other) {
		final T container = (T) other;
		return plus(container);
	}
	
	/**
	 * Subtracts <code>other</code> from <code>this</code>, element-wise.
	 * 
	 * @param other to be subtracted from <code>this</code>
	 * @return <code>(this - other)</code> for each element
	 */
	public T minus(final T other) {
		return elementOperation(other, SUBTRACT);
	}
	
	/**
	 * Computes the hadamard product (element-wise multiplication) of <code>this</code> and <code>other</code>.
	 * 
	 * @param other to be multiplied to <code>this</code>, element-wise
	 * @return <code>(this * other)</code> for each element
	 */
	public T hadamard(final T other) {
		return elementOperation(other, MULTIPLY);
	}
	
	/**
	 * Divides <code>this</code> by <code>other</code>, element-wise.
	 * 
	 * @param other to be divided by
	 * @return <code>(this / other)</code> for each element
	 */
	public T elementDivide(final T other) {
		return elementOperation(other, DIVIDE);
	}
	
	/**
	 * Scales this tensor by some amount. Multiplies every element by the given value and returns a new
	 * ElementContainer with the result.
	 * 
	 * @param value scale factor
	 * @return new ElementContainer scaled by <code>value</code>
	 */
	public T scale(final double value) {
		final DoubleApplier scaler = d -> d * value;
		
		return transform(scaler);
	}
}
