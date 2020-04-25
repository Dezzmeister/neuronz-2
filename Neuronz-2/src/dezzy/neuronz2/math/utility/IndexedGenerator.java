package dezzy.neuronz2.math.utility;

/**
 * Defines a function that can generate a value for a tensor of any rank, given the indices
 * into the tensor.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface IndexedGenerator {
	
	/**
	 * Returns a new value based on the indices into a tensor of any rank. This can include a vector
	 * (in which only <code>indices[0]</code> is used), a matrix, or any other tensor.
	 * 
	 * @param indices index array; this has a different length depending on the type of tensor being generated
	 * @return a value for the location in the tensor specified by <code>indices</code>
	 */
	double generate(final int ... indices);
}
