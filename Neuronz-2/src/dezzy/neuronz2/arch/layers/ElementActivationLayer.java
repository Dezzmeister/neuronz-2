package dezzy.neuronz2.arch.layers;

import java.util.HashMap;
import java.util.Map;

import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.FuncDerivPair;

/**
 * An activation layer in a neural network, applies an activation function
 * element-wise to the input.
 *
 * @author Joe Desmond
 * @param <T> type of the input to this activation layer (i.e.; matrix, vector, etc.)
 */
public class ElementActivationLayer<T extends ElementContainer<T>> implements ParallelLayer<T, T> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8370778816266631023L;

	/**
	 * The activation function and its derivative
	 */
	private final FuncDerivPair activationFunction;
	
	/**
	 * The latest input to this layer. May be used in backpropagation
	 * 
	 * TODO: Implement this
	 */
	@SuppressWarnings("unused")
	private T latestInput;
	
	/**
	 * The latest output of this layer. Used in backpropagation
	 */
	private T latestOutput;
	
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
	 * <code>prevActivations</code>. Saves the application of the activation function internally
	 * as well as the input to the function (for backpropagation). Returns the application
	 * of the activation function on <code>prevActivations</code>.
	 * 
	 * 
	 * @param prevActivations output from the previous layer
	 * @return output from this layer
	 */
	@Override
	public T forwardPass(final T prevActivations) {
		latestInput = prevActivations;
		latestOutput = prevActivations.transform(activationFunction.function);
		return latestOutput;
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
		final T outputInputDeriv = latestOutput.transform(activationFunction.derivative);
		
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
	
	@Override
	public int sublayers() {
		return 1;
	}
	
	/**
	 * Returns zero because there are no learnable parameters in this layer.
	 * 
	 * @return zero
	 */
	@Override
	public int parameterCount() {
		return 0;
	}

	@Override
	public ParallelForwardPass<T> parallelForwardPass(final T prevActivations) {
		final Map<Layer<?, ?>, ElementContainer<?>> latestInputs = new HashMap<>();
		final Map<Layer<?, ?>, ElementContainer<?>> latestOutputs = new HashMap<>();
		
		latestInputs.put(this, prevActivations);
		
		final T output = prevActivations.transform(activationFunction.function);
		
		latestOutputs.put(this, output);
		
		return new ParallelForwardPass<>(output, latestInputs, latestOutputs);
	}

	@Override
	public ParallelBackwardPass<T> parallelBackprop(final ParallelForwardPass<T> prevForward, final T errorOutputDeriv, final boolean isFirstLayer) {
		@SuppressWarnings("unchecked")
		final T prevLatestOutput = (T) prevForward.latestOutputs.get(this);
		
		final T outputInputDeriv = prevLatestOutput.transform(activationFunction.derivative);
		
		final T output = errorOutputDeriv.hadamard(outputInputDeriv);
		
		return new ParallelBackwardPass<>(output, Map.of());
	}

	@Override
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, double learningRate) {
		// TODO Auto-generated method stub
		
	}
}
