package main.math.utility;

import main.math.constructs.Vector;

/**
 * Utility functions for Vectors that do not belong in the {@link Vector} class.
 *
 * @author Joe Desmond
 */
public final class VectorUtils {
	
	/**
	 * Calculates the cross product of two 3-component Vectors.
	 * 
	 * @param a a Vector with 3 components
	 * @param b a Vector with 3 components
	 * @return a x b
	 */
	public static final Vector cross(final Vector a, final Vector b) {
		if (a.dimension != 3 || b.dimension != 3) {
			throw new DimensionMismatchException("Cross product can only be calculated for Vectors with 3 components!");
		}
		
		final float x = a.get(1) * b.get(2) - a.get(2) * b.get(1);
		final float y = a.get(2) * b.get(0) - a.get(0) * b.get(2);
		final float z = a.get(0) * b.get(1) - a.get(1) * b.get(0);
		
		return new Vector(x, y, z);
	}
}
