package dezzy.neuronz2.arch.layers;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.FuncDerivPair;

/**
 * An activation layer in a neural network, applies an activation function
 * element-wise to the input.
 *
 * @author Joe Desmond
 * @param <T> type of the input to this activation layer (i.e.; matrix, vector, etc.)
 */
public class ElementActivationLayer<T extends ElementContainer<T>> implements Layer<T, T> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8370778816266631023L;

	/**
	 * The activation function and its derivative
	 */
	private final FuncDerivPair activationFunction;
	
	/**
	 * The latest input of this layer. Used in backpropagation
	 */
	private T latestInput;
	
	/**
	 * Constructs the activation layer given an activation function and its derivative.
	 * 
	 * @param _activationFunction the activation function and its derivative
	 */
	public ElementActivationLayer(final FuncDerivPair _activationFunction) {
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
	public T forwardPass(final T prevActivations) {
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
	 * @param isFirstLayer unused
	 * @return derivative of the network error with respect to this layer's input
	 */
	@Override
	public T backprop(final T errorOutputDeriv, final boolean isFirstLayer) {
		final T outputInputDeriv = latestInput.transform(activationFunction.derivative);
		
		return errorOutputDeriv.hadamard(outputInputDeriv);
	}
	
	/**
	 * Not implemented: there are no weights in this layer.
	 * 
	 * @param learningRate unused
	 */
	@Override
	public void update(final double learningRate) {
		
	}
}
