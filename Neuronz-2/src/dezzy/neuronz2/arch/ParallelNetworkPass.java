package dezzy.neuronz2.arch;

import java.util.concurrent.Callable;

import dezzy.neuronz2.arch.error.CompleteErrorFunc;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A single forward and backward pass through a layered network. Returns an object mapping layers to their
 * parameter gradients for the pass.
 * <p>
 * This class and the Parallel Layer framework allow several runs to be performed on a single network simultaneously.
 * After these runs complete, the gradients can be summed and averaged to update the layer.
 *
 * @author Joe Desmond
 * @param <I> input tensor type
 * @param <O> output tensor type
 */
public class ParallelNetworkPass<I extends ElementContainer<I>, O extends ElementContainer<O>> implements Callable<ParallelBackwardPass<I>> {
	
	/**
	 * The layered network
	 */
	public final ParallelLayer<I, O> layer;
	
	/**
	 * Input for the forward pass
	 */
	public final I input;
	
	/**
	 * Expected output of the forward pass
	 */
	public final O expectedOutput;
	
	/**
	 * Error function to use for calculating gradients
	 */
	public final CompleteErrorFunc<O> errorFunc;
	
	/**
	 * The actual output of the network, which is stored here after the network runs
	 */
	public O actualOutput;
	
	/**
	 * Constructs an object containing data to calculate a single forward pass and a single backward pass through a parallel
	 * layer network.
	 * 
	 * @param _layer parallel layer network
	 * @param _input input for the forward pass
	 * @param _expectedOutput expected output of the forward pass
	 * @param _errorFunc error function to use for backward pass
	 */
	public ParallelNetworkPass(final ParallelLayer<I, O> _layer, final I _input, final O _expectedOutput, final CompleteErrorFunc<O> _errorFunc) {
		layer = _layer;
		input = _input;
		expectedOutput = _expectedOutput;
		errorFunc = _errorFunc;
	}
	
	@Override
	public ParallelBackwardPass<I> call() throws Exception {
		final ParallelForwardPass<O> forwardPass = layer.parallelForwardPass(input);
		actualOutput = forwardPass.output;
		final double error = errorFunc.errorFunction.condense(expectedOutput, forwardPass.output);
		final O errorDeriv = errorFunc.errorFunctionDerivative.calculate(expectedOutput, forwardPass.output, error);
		
		final ParallelBackwardPass<I> backwardPass = layer.parallelBackprop(forwardPass, errorDeriv, true);
		
		return backwardPass;
	}
	
}
