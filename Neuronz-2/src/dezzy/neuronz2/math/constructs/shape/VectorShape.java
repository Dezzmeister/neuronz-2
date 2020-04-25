package dezzy.neuronz2.math.constructs.shape;

import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * A class containing the shape of a vector. A vector is only defined by its
 * length (number of components), so this class has one member.
 *
 * @author Joe Desmond
 */
public class VectorShape extends Shape<Vector> {
	
	/**
	 * Number of components in the vector
	 */
	public final int length;
	
	/**
	 * Constructs a shape object describing a vector with the given shape. Does
	 * NOT create a vector, that is done with {@link #generate(IndexedGenerator)}.
	 * 
	 * @param _length number of components
	 */
	public VectorShape(final int _length) {
		length = _length;
	}
	
	@Override
	public Vector generate(final IndexedGenerator generator) {
		return Vector.generate(generator, length);
	}	
}
