package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Defines a way to convert between a tensor from a convolutional neural network
 * to a vector for a fully connected neural network. 
 * <p>
 * Classes that implement this interface must define {@link #flatten(Tensor3)} 
 * and {@link #deflatten(Vector)} so that the two are inverses of each other:
 * if <code>flatten(t)</code> produces a vector <b>v</b>, then <code>deflatten(v)</code>
 * must produce a copy of the original tensor <b>t</b>.
 *
 * @author Joe Desmond
 */
public interface Flattener extends Layer<Tensor3, Vector> {
	
	/**
	 * Converts a rank 3 tensor into a vector. Undoes {@link #deflatten(Vector)}.
	 * 
	 * @param tensor 3D tensor
	 * @return 1D vector
	 */
	public Vector flatten(final Tensor3 tensor);
	
	/**
	 * Converts a vector into a rank 3 tensor. Undoes {@link #flatten(Tensor3)}.
	 * 
	 * @param vector 1D vector
	 * @return 3D tensor
	 */
	public Tensor3 deflatten(final Vector vector);
}
