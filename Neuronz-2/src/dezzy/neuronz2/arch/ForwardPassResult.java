package dezzy.neuronz2.arch;

import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * The result of one forward pass through a network.
 *
 * @author Joe Desmond
 * @param <T> tensor type (data returned from final layer in the network)
 */
public class ForwardPassResult<T extends ElementContainer<T>> {
	
	/**
	 * Data obtained from the last layer in the network
	 */
	public final T actualOutput;
	
	/**
	 * Error calculated by some cost function given the actual output and some expected output
	 */
	public final double error;
	
	/**
	 * Constructs a forward pass result with the given network output and calculated network error.
	 * 
	 * @param _actualOutput network output
	 * @param _error network error
	 */
	public ForwardPassResult(final T _actualOutput, final double _error) {
		actualOutput = _actualOutput;
		error = _error;
	}
}
