package dezzy.neuronz2.arch.layers;

import java.util.HashMap;
import java.util.Map;

import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.TensorFunction;

/**
 * An activation layer using a tensor-valued activation function. Unlike {@link ElementActivationLayer}
 * the activation function is not applied element-wise, it is applied to the entire input at once.
 *
 * @author Joe Desmond
 * @param <T> tensor type (vector, matrix, etc.)
 */
public class TensorActivationLayer<T extends ElementContainer<T>> implements ParallelLayer<T, T> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -962273503392173336L;
	
	/**
	 * The activation function and its derivative
	 */
	private final TensorFunction<T> activationFunction;
	
	/**
	 * The latest input to this layer
	 */
	private T latestInput;
	
	/**
	 * Constructs a tensor activation layer from the given tensor-valued activation function.
	 * 
	 * @param _activationFunction activation function
	 */
	public TensorActivationLayer(final TensorFunction<T> _activationFunction) {
		activationFunction = _activationFunction;
	}

	/**
	 * Saves the input internally for use in backpropagation, and returns the result
	 * of {@linkplain #activationFunction this} activation function applied to the input.
	 * 
	 * @param prevActivations input tensor to this layer
	 * @return output tensor of this layer
	 */
	@Override
	public T forwardPass(final T prevActivations) {
		latestInput = prevActivations;
		return activationFunction.function.apply(prevActivations);
	}
	
	/**
	 * Computes the derivative of this layer's output with respect to its input by applying the
	 * derivative of the {@linkplain #activationFunction activation function} to the 
	 * {@linkplain #latestInput latest input}, then multiplying that derivative element-wise
	 * with the derivative of the network error with respect to this layer's output.
	 * 
	 * @param errorOutputDeriv (partial) derivative of the network error with respect to this layer's output
	 * @param isFirstLayer unused
	 * @return (partial) derivative of the network error with respect to this layer's input 
	 */
	@Override
	public T backprop(final T errorOutputDeriv, final boolean isFirstLayer) {
		final T outputInputDeriv = activationFunction.derivative.apply(latestInput);
		
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
	
	/**
	 * Returns zero because there are no learnable parameters in this layer.
	 * 
	 * @return zero
	 */
	@Override
	public int parameterCount() {
		return 0;
	}
	
	/**
	 * Returns one because this layer is not composed of any sublayers.
	 * 
	 * @return one
	 */
	@Override
	public int sublayers() {
		return 1;
	}

	@Override
	public ParallelForwardPass<T> parallelForwardPass(final T prevActivations) {
		final Map<Layer<?, ?>, ElementContainer<?>> latestInputs = new HashMap<>();
		
		latestInputs.put(this, prevActivations);
		
		final T output = activationFunction.function.apply(prevActivations);
		
		return new ParallelForwardPass<>(output, latestInputs, Map.of());
	}

	@Override
	public ParallelBackwardPass<T> parallelBackprop(final ParallelForwardPass<T> prevForward, final T errorOutputDeriv, final boolean isFirstLayer) {
		@SuppressWarnings("unchecked")
		final T prevLatestInput = (T) prevForward.latestInputs.get(this);
		
		final T outputInputDeriv = activationFunction.derivative.apply(prevLatestInput);
		
		return new ParallelBackwardPass<>(errorOutputDeriv.hadamard(outputInputDeriv), Map.of());
	}

	@Override
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, final double learningRate) {
		// TODO Auto-generated method stub
		
	}
}
