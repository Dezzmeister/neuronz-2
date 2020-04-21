package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * An activation layer in a convolutional neural network, applies an activation function
 * element-wise to every feature map.
 *
 * @author Joe Desmond
 */
public class ActivationLayer implements Layer {
	
	/**
	 * The activation function and its derivative
	 */
	private final FuncDerivPair activationFunction;
	
	/**
	 * The latest input of this layer. Used in backpropagation
	 */
	private Tensor3 latestInput;
	
	/**
	 * Constructs the activation layer given an activation function and its derivative.
	 * 
	 * @param _activationFunction the activation function and its derivative
	 */
	public ActivationLayer(final FuncDerivPair _activationFunction) {
		activationFunction = _activationFunction;
	}
	
	/**
	 * Applies {@linkplain #activationFunction this} activation function element-wise to
	 * <code>prevActivations</code>. Saves <code>prevActivations</code> internally for backpropagation, 
	 * and returns the application of the activation function to the input.
	 * 
	 * @param prevActivations output from the previous layer
	 * @return output from this layer
	 */
	@Override
	public Tensor3 activations(final Tensor3 prevActivations) {
		latestInput = prevActivations;
		return prevActivations.transform(activationFunction.function);
	}

	/**
	 * First computes the derivative of this layer's output with respect to its input; given
	 * by applying {@linkplain #activationFunction this} activation function's derivative element-wise to
	 * the {@linkplain #latestOutput previous output} of forward-propagation. Multiplies this derivative
	 * element-wise with the given derivative (chain rule), and returns this new derivative.
	 * 
	 * @param errorOutputDeriv derivative of the network error with respect to this layer's output
	 * @return derivative of the network error with respect to this layer's input
	 */
	@Override
	public Tensor3 backprop(final Tensor3 errorOutputDeriv) {
		final Tensor3 outputInputDeriv = latestInput.transform(activationFunction.derivative);
		
		return errorOutputDeriv.hadamard(outputInputDeriv);
	}
}
