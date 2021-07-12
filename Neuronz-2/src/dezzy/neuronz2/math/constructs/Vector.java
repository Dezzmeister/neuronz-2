package dezzy.neuronz2.math.constructs;

import dezzy.neuronz2.math.constructs.shape.VectorShape;
import dezzy.neuronz2.math.utility.DimensionMismatchException;
import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Represents a Vector with any number of elements. Vectors should not integrate with the JCuda API - 
 * see {@link dezzy.neuronz2.cuda.math.GVector GVector} for a Vector usable with JCuda.
 *
 * @author Joe Desmond
 */
public class Vector extends ElementContainer<Vector> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4107315004380260571L;

	/**
	 * The components of the Vector, these should not change.
	 */
	protected final double[] components;
	
	/**
	 * The length of the Vector in space, calculated upon construction
	 */
	public final double length;
	
	/**
	 * The number of components in this Vector
	 */
	public final int dimension;
	
	/**
	 * Creates a Vector with the given component values.
	 * 
	 * @param _components the components of this vector
	 */
	public Vector(final double ... _components) {
		components = _components;
		dimension = components.length;
		length = calculateLength();
	}
	
	/**
	 * Protected copy constructor. Useful for subclasses to provide ways to construct themselves
	 * from existing Vectors. A reference to the other vector's components is copied because it is
	 * assumed that the components are immutable.
	 * 
	 * @param other vector to copy
	 */
	protected Vector(final Vector other) {
		components = getComponents(other);
		dimension = components.length;
		length = other.length;
	}
	
	/**
	 * Returns the shape of this vector (the number of components in the vector).
	 * 
	 * @return shape of this vector
	 */
	public VectorShape shape() {
		return new VectorShape(dimension);
	}
	
	/**
	 * Gets the components array of another vector. Useful for modifying another vector's components
	 * from within a subclass.
	 * 
	 * @param other other vector
	 * @return components of <code>other</code>
	 */
	protected double[] getComponents(final Vector other) {
		return other.components;
	}
	
	/**
	 * Generates a vector with the specified length using the given generator function. Calls
	 * {@link IndexedGenerator#generate(int...) generator.generate(i)} with every component index
	 * to obtain the values of the vector.
	 * 
	 * @param generator generator function, used to generate components of the vector
	 * @param length number of components in the vector
	 * @return a new vector
	 */
	public static Vector generate(final IndexedGenerator generator, final int length) {
		final double[] out = new double[length];
		
		for (int i = 0; i < length; i++) {
			out[i] = generator.generate(i);
		}
		
		return new Vector(out);
	}
	
	/**
	 * Calculates the inner (dot) product of this Vector and another. Both Vectors should have the same number of components. <br>
	 * Throws a {@link DimensionMismatchException} if <code>other</code> does not have the same number of components as this Vector.
	 * 
	 * @param other other Vector
	 * @return the dot product of this Vector and <code>other</code>
	 */
	public final double innerProduct(final Vector other) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Vectors must have an equal number of components to calculate an inner product!");
		}
		
		double sum = 0;
		for (int i = 0; i < components.length; i++) {
			sum += components[i] * other.components[i];
		}
		
		return sum;
	}
	
	/**
	 * Calculates the outer (tensor) product of this Vector and another.
	 * 
	 * @param other other Vector
	 * @return the tensor product of this Vector and <code>other</code>
	 */
	public final Matrix outerProduct(final Vector other) {
		final double[][] result = new double[dimension][other.dimension];
		
		for (int row = 0; row < result.length; row++) {
			for (int col = 0; col < result[0].length; col++) {
				result[row][col] = components[row] * other.components[col];
			}
		}
		
		return new Matrix(result);
	}
	
	/**
	 * Returns the value of the component at the given index of this Vector. <br>
	 * <b>This function does not check if <code>index</code> is within an acceptable range, an exception will be thrown if <code>index</code>
	 * is negative or is greater than or equal to the length of this Vector!</b>
	 * 
	 * @param index index of the component in this Vector
	 * @return value of the component
	 */
	public final double get(final int index) {
		return components[index];
	}
	
	/**
	 * Copies the components of this vector to a double array. Internally uses 
	 * {@link System#arraycopy(src, srcPos, dest, destPos, length)}, where <code>src</code>
	 * is this vector's components array.
	 * 
	 * @param srcIndex starting index (in this vector) to copy from
	 * @param dest destination array
	 * @param destIndex destination index (into <code>dest</code>) to copy to
	 * @param length number of elements to copy
	 */
	public final void copyTo(final int srcIndex, final double[] dest, final int destIndex, final int length) {
		System.arraycopy(components, srcIndex, dest, destIndex, length);
	}
	
	/**
	 * Calculates the length of this Vector using the Pythagorean Theorem.
	 * 
	 * @return the length of this Vector
	 */
	private final double calculateLength() {
		double sum = 0;
		for (int i = 0; i < components.length; i++) {
			sum += (components[i] * components[i]);
		}
		
		return (double) Math.sqrt(sum);
	}
	
	/**
	 * Returns a new Vector with the specified value appended as an additional component.
	 * 
	 * @param value value to be appended
	 * @return a new Vector, 1 dimension higher than this Vector
	 */
	public final Vector append(final double value) {
		final double[] result = new double[dimension + 1];
		
		for (int i = 0; i < dimension; i++) {
			result[i] = components[i];
		}
		
		result[dimension] = value;
		
		return new Vector(result);
	}
	
	/**
	 * Appends another Vector to this one.
	 * 
	 * @param other vector to be appended
	 * @return a new Vector with dimension <code>(this.dimension + other.dimension)</code>
	 */
	public final Vector append(final Vector other) {
		final double[] result = new double[dimension + other.dimension];
		
		for (int i = 0; i < dimension; i++) {
			result[i] = components[i];
		}
		
		for (int i = dimension; i < result.length; i++) {
			result[i] = other.components[i - dimension];
		}
		
		return new Vector(result);
	}
	
	/**
	 * Returns a copy of this Vector with the last element removed.
	 * 
	 * @return this Vector without the last element
	 */
	public final Vector removeLastElement() {
		final double[] result = new double[dimension - 1];
		
		System.arraycopy(components, 0, result, 0, dimension - 1);
		
		return new Vector(result);
	}
	
	/**
	 * Returns the Vector starting at the start index (inclusive) and ending at the end index (non-inclusive).
	 * 
	 * @param start index of first component (inclusive)
	 * @param end index of last component (non-inclusive)
	 * @return the vector from [start, end) in this vector
	 */
	public final Vector trim(final int start, final int end) {
		final double[] result = new double[end - start];
		
		System.arraycopy(components, start, result, 0, result.length);
		
		return new Vector(result);
	}
	
	/**
	 * Sums the components of this vector and returns the result.
	 * 
	 * @return the sum of every component in this vector
	 */
	public final double sum() {
		double sum = 0;
		
		for (int i = 0; i < components.length; i++) {
			sum += components[i];
		}
		
		return sum;
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
	public final Vector elementOperation(final Vector other, final DoubleOperator operator) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Vectors must have same dimensions to perform element operations!");
		}
		
		final double[] result = new double[dimension];
		for (int i = 0; i < dimension; i++) {
			result[i] = operator.operate(components[i], other.components[i]);
		}
		
		return new Vector(result);
	}
	
	@Override
	public final Vector transform(final DoubleApplier operator) {
		final double[] result = new double[dimension];
		
		for (int i = 0; i < components.length; i++) {
			result[i] = operator.apply(components[i]);
		}
		
		return new Vector(result);
	}
}
