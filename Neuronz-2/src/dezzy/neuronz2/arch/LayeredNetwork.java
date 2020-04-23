package dezzy.neuronz2.arch;

import dezzy.neuronz2.arch.error.CompleteErrorFunc;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * Part of the new neural network architecture; a network composed of one or more layers.
 * The {@link #network} field may internally contain several layers, but it is treated as 
 * one layer because it is assumed that calling the specified functions in {@link Layer}
 * on {@link #network} will work as expected, regardless of implementation.
 *
 * @author Joe Desmond
 * @param <I> input tensor type to the network (vector, matrix, etc.)
 * @param <O> output tensor type from the network (vector, matrix, etc.)
 */
public class LayeredNetwork<I extends ElementContainer<I>, O extends ElementContainer<O>> {
	
	/**
	 * The actual network; can be either one layer or a 
	 * {@linkplain dezzy.neuronz2.arch.layers.LayerSequence sequence of layers}
	 */
	private final Layer<I, O> network;
	
	/**
	 * The error/cost function and its derivative with respect to the network output
	 */
	private final CompleteErrorFunc<O> errorFunc;
	
	/**
	 * Constructs a network with the given layer(s) and cost function.
	 * 
	 * @param _network layer(s)
	 * @param _errorFunc cost function
	 */
	public LayeredNetwork(final Layer<I, O> _network, final CompleteErrorFunc<O> _errorFunc) {
		network = _network;
		errorFunc = _errorFunc;
	}
	
	/**
	 * Propagates the input through the network and calculates the error given the expected output.
	 * This function may change the state of the {@linkplain #network network}; particularly if any of the
	 * layers have weights.
	 *  
	 * @param input input to the network
	 * @param expectedOutput expected output of the network
	 * @return actual output and error given expected output
	 */
	public final ForwardPassResult<O> forwardPass(final I input, final O expectedOutput) {
		final O actualOutput = network.forwardPass(input);
		final double error = errorFunc.errorFunction.condense(expectedOutput, actualOutput);
		
		return new ForwardPassResult<O>(actualOutput, error);
	}
	
	/**
	 * Propagates the error backwards through the network. Does not update weights; although internal
	 * weight deltas may be updated.
	 * <p>
	 * <b>NOTE:</b> The arguments to this function should be those obtained from the most recent call
	 * to {@link #forwardPass(ElementContainer, ElementContainer) forwardPass()}.
	 * 
	 * @param expectedOutput expected output of the previous forward pass
	 * @param actualOutput actual output of the previous forward pass
	 * @param error error of the previous forward pass 
	 */
	public final void backprop(final O expectedOutput, final O actualOutput, final double error) {
		final O errorDeriv = errorFunc.errorFunctionDerivative.calculate(expectedOutput, actualOutput, error);
		network.backprop(errorDeriv, true);
	}
	
	/**
	 * Updates the learnable parameters in the network.
	 * 
	 * @param learningRate learning rate
	 */
	public final void update(final double learningRate) {
		network.update(learningRate);
	}
}
