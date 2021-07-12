package dezzy.neuronz2.cuda.math;

import dezzy.neuronz2.math.constructs.Vector;
import jcuda.Pointer;

/**
 * An extension of the Vector class for vectors that are used with the JCuda API. The JCuda methods
 * in this class could have been members of the Vector class, but they are included here instead
 * to keep the purpose of Vector clear.
 * 
 * @author Joe Desmond
 */
public class GVector extends Vector {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3693566986181941464L;
	
	/**
	 * Promotes the input vector to a GVector by copying its members.
	 * 
	 * @param base
	 */
	public GVector(final Vector base) {
		// Invoke the protected copy constructor to copy base's components and promote base
		// to a GVector
		super(base);
	}
	
	/**
	 * Returns a pointer to the components of this vector.
	 * 
	 * @return pointer to the components of this vector
	 */
	public Pointer getPointer() {
		return Pointer.to(components);
	}
}
