package dezzy.neuronz2.math.constructs;

import java.io.Serializable;

import dezzy.neuronz2.math.utility.TensorApplier;

/**
 * A tensor-valued function and its derivative with respect to the input.
 *
 * @author Joe Desmond
 * @param <T> tensor type
 * @see {@link TensorApplier}
 */
public class TensorFunction<T extends ElementContainer<T>> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3196210006381065111L;
	
	/**
	 * The function itself; takes a tensor as an argument and returns a tensor of the same rank
	 */
	public final TensorApplier<T> function;
	
	/**
	 * The derivative of <code>function</code> with respect to its input
	 */
	public final TensorApplier<T> derivative;
	
	/**
	 * Constructs a tensor function-derivative pair from the given functions.
	 * 
	 * @param _function function
	 * @param _derivative derivative of <code>_function</code> with respect to its input
	 */
	public TensorFunction(final TensorApplier<T> _function, final TensorApplier<T> _derivative) {
		function = _function;
		derivative = _derivative;
	}	
}
