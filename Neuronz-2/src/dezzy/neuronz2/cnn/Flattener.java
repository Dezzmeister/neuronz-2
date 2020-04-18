package dezzy.neuronz2.cnn;

import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Defines a way to convert between a matrix from a convolutional neural network
 * to a vector for a fully connected neural network. 
 * <p>
 * Classes that implement this interface must define {@link #flatten(Matrix)} 
 * and {@link #deflatten(Vector)} so that the two are inverses of each other:
 * if <code>flatten(m)</code> produces a vector <b>v</b>, then <code>deflatten(v)</code>
 * must produce a copy of the original matrix <b>m</b>.
 *
 * @author Joe Desmond
 */
public interface Flattener {
	
	/**
	 * Converts a Matrix into a Vector. Undoes {@link #deflatten(Vector)}.
	 * 
	 * @param matrix 2D matrix
	 * @return 1D vector
	 */
	public Vector flatten(final Matrix matrix);
	
	/**
	 * Converts a Vector into a Matrix. Undoes {@link #flatten(Matrix)}.
	 * 
	 * @param vector 1D vector
	 * @return 2D matrix
	 */
	public Matrix deflatten(final Vector vector);
}
