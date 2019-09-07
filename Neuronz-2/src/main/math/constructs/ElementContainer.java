package main.math.constructs;

import main.math.utility.FloatApplier;
import main.math.utility.FloatOperator;

/**
 * Represents anything that contains a collection of float elements.
 *
 * @author Joe Desmond
 */
public interface ElementContainer<T> {
	
	/**
	 * Performs an operation with each element of <code>this</code> and <code>other</code>.
	 * 
	 * @param other something containing float elements
	 * @param operator operation to be performed on each pair of elements
	 * @return the application of <code>operator</code> on <code>this</code> and <code>other</code>
	 */
	T elementOperation(final T other, final FloatOperator operator);
	
	/**
	 * Performs an operation on each element of <code>this</code>.
	 * 
	 * @param transformation operation to be performed on each element
	 * @return the application of <code>operator</code> on <code>this</code>
	 */
	T transform(final FloatApplier transformation);
}
