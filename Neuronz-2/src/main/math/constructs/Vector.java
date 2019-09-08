package main.math.constructs;

import main.math.utility.DimensionMismatchException;
import main.math.utility.FloatApplier;
import main.math.utility.FloatOperator;

/**
 * Represents a Vector with any number of elements.
 *
 * @author Joe Desmond
 */
public final class Vector extends ElementContainer<Vector> {
	/**
	 * The components of the Vector, these should not change.
	 */
	private final float[] components;
	
	/**
	 * The length of the Vector in space, calculated upon construction
	 */
	public final float length;
	
	/**
	 * The number of components in this Vector
	 */
	public final int dimension;
	
	/**
	 * Creates a Vector with the given component values.
	 * 
	 * @param _components the components of this vector
	 */
	public Vector(final float ... _components) {
		components = _components;
		dimension = components.length;
		length = calculateLength();
	}
	
	/**
	 * Calculates the inner (dot) product of this Vector and another. Both Vectors should have the same number of components. <br>
	 * Throws a {@link DimensionMismatchException} if <code>other</code> does not have the same number of components as this Vector.
	 * 
	 * @param other other Vector
	 * @return the dot product of this Vector and <code>other</code>
	 */
	public final float innerProduct(final Vector other) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Vectors must have an equal number of components to calculate an inner product!");
		}
		
		float sum = 0;
		for (int i = 0; i < components.length; i++) {
			sum += components[i] * other.components[i];
		}
		
		return sum;
	}
	
	/**
	 * Returns the value of the component at the given index of this Vector. <br>
	 * <b>This function does not check if <code>index</code> is within an acceptable range, an exception will be thrown if <code>index</code>
	 * is negative or is greater than or equal to the length of this Vector!</b>
	 * 
	 * @param index index of the component in this Vector
	 * @return value of the component
	 */
	public final float get(final int index) {
		return components[index];
	}
	
	/**
	 * Calculates the length of this Vector using the Pythagorean Theorem.
	 * 
	 * @return the length of this Vector
	 */
	private final float calculateLength() {
		float sum = 0;
		for (int i = 0; i < components.length; i++) {
			sum += (components[i] * components[i]);
		}
		
		return (float) Math.sqrt(sum);
	}
	
	/**
	 * Returns a new Vector with the specified value appended as an additional component.
	 * 
	 * @param value value to be appended
	 * @return a new Vector, 1 dimension higher than this Vector
	 */
	public final Vector append(final float value) {
		final float[] result = new float[dimension + 1];
		
		for (int i = 0; i < dimension; i++) {
			result[i] = components[i];
		}
		
		result[dimension] = value;
		
		return new Vector(result);
	}
	
	/**
	 * Converts this Vector into an easily readable format.
	 */
	@Override
	public String toString() {
		String out = "[";
		
		for (int i = 0; i < components.length - 1; i++) {
			out += components[i] + ", ";
		}
		
		return out + components[components.length - 1] + "]";
	}
	
	/**
	 * Applies an operation to each element of this Vector and another. Throws a {@link DimensionMismatchException} if the dimension of this Vector
	 * does not match the dimension of <code>other</code>. Elements from this Vector are passed in as <code>a</code>, and elements from the other Vector
	 * are passed in as <code>b</code>.
	 * 
	 * @param other other Vector
	 * @param operator operation to be applied
	 * @return the result of the operation
	 */
	@Override
	public final Vector elementOperation(final Vector other, final FloatOperator operator) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Vectors must have same dimensions to perform element operations!");
		}
		
		final float[] result = new float[dimension];
		for (int i = 0; i < dimension; i++) {
			result[i] = operator.operate(components[i], other.components[i]);
		}
		
		return new Vector(result);
	}
	
	@Override
	public final Vector transform(final FloatApplier operator) {
		final float[] result = new float[dimension];
		
		for (int i = 0; i < components.length; i++) {
			result[i] = operator.apply(components[i]);
		}
		
		return new Vector(result);
	}
}
