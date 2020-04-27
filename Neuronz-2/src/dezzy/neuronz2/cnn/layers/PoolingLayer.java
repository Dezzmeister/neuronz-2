package dezzy.neuronz2.cnn.layers;

import java.util.HashMap;
import java.util.Map;

import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * A pooling layer in a convolutional neural network.
 *
 * @author Joe Desmond
 */
public class PoolingLayer implements ParallelLayer<Tensor3, Tensor3> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8305938005765678674L;

	/**
	 * The pooling operation
	 */
	private final PoolingOperation poolingOperation;
	
	/**
	 * The number of rows in the pooling window
	 */
	private final int windowRows;
	
	/**
	 * The number of columns in the pooling window
	 */
	private final int windowCols;
	
	/**
	 * The pooling window row stride
	 */
	private final int rowStride;
	
	/**
	 * The pooling window column stride
	 */
	private final int colStride;
	
	/**
	 * The latest input to the pooling operation, used in backpropagation
	 */
	private Tensor3 latestInput;
	
	/**
	 * Constructs a pooling layer with the given pooling operation and pooling window size.
	 * 
	 * @param _poolingOperation pooling operation (Example: {@link PoolingOperation#MAX_POOLING})
	 * @param _windowRows number of rows in the pooling window
	 * @param _windowCols number of columns in the pooling window
	 */
	public PoolingLayer(final PoolingOperation _poolingOperation, final int _windowRows, final int _windowCols, final int _rowStride, final int _colStride) {
		poolingOperation = _poolingOperation;
		windowRows = _windowRows;
		windowCols = _windowCols;
		rowStride = _rowStride;
		colStride = _colStride;
	}
	
	/**
	 * Applies {@linkplain #poolingOperation this} pooling operation layer-wise to the given activations and returns a smaller tensor
	 * (same number of layers; but smaller layers) for the next network layer. Saves the input tensor to be used for backpropagation.
	 * 
	 * @param prevActivations output of previous network layer
	 * @return output of this network layer
	 */
	@Override
	public Tensor3 forwardPass(final Tensor3 prevActivations) {
		final Matrix[] output = new Matrix[prevActivations.dimension];
		
		for (int i = 0; i < prevActivations.dimension; i++) {
			final Matrix result = prevActivations.getLayer(i).poolingTransform(windowRows, windowCols, rowStride, colStride, poolingOperation);
			
			output[i] = result;
		}
		
		latestInput = prevActivations;
		
		return new Tensor3(output);
	}
	
	/**
	 * Calculates the derivative of the error with respect to the input using 
	 * {@link PoolingOperation#backprop(Matrix, Matrix, int, int) poolingOperation.backprop()}.
	 * 
	 * @param errorOutputDeriv the (partial) derivative of the network error with respect to the output of this layer
	 * @param isFirstLayer unused
	 * @return the (partial) derivative of the network error with respect to the input to this layer
	 */
	@Override
	public Tensor3 backprop(final Tensor3 errorOutputDeriv, final boolean isFirstLayer) {
		final Matrix[] errorInputDeriv = new Matrix[errorOutputDeriv.dimension];
		
		for (int m = 0; m < errorOutputDeriv.dimension; m++) {
			final Matrix pooled = errorOutputDeriv.getLayer(m);
			final Matrix input = latestInput.getLayer(m);
			
			errorInputDeriv[m] = poolingOperation.backprop(input, pooled, windowRows, windowCols, rowStride, colStride);
		}
		
		return new Tensor3(errorInputDeriv);
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
	public ParallelForwardPass<Tensor3> parallelForwardPass(final Tensor3 prevActivations) {
		final Map<Layer<?, ?>, ElementContainer<?>> latestInputs = new HashMap<>();
		
		final Matrix[] output = new Matrix[prevActivations.dimension];
		
		for (int i = 0; i < prevActivations.dimension; i++) {
			final Matrix result = prevActivations.getLayer(i).poolingTransform(windowRows, windowCols, rowStride, colStride, poolingOperation);
			
			output[i] = result;
		}
		
		latestInputs.put(this, prevActivations);
		
		return new ParallelForwardPass<>(new Tensor3(output), latestInputs, Map.of());
	}

	@Override
	public ParallelBackwardPass<Tensor3> parallelBackprop(final ParallelForwardPass<Tensor3> prevForward, final Tensor3 errorOutputDeriv, final boolean isFirstLayer) {
		final Tensor3 prevLatestInput = (Tensor3) prevForward.latestInputs.get(this);
		
		final Matrix[] errorInputDeriv = new Matrix[errorOutputDeriv.dimension];
		
		for (int m = 0; m < errorOutputDeriv.dimension; m++) {
			final Matrix pooled = errorOutputDeriv.getLayer(m);
			final Matrix input = prevLatestInput.getLayer(m);
			
			errorInputDeriv[m] = poolingOperation.backprop(input, pooled, windowRows, windowCols, rowStride, colStride);
		}
		
		return new ParallelBackwardPass<>(new Tensor3(errorInputDeriv), Map.of());
	}

	@Override
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, final double learningRate) {
		// TODO Auto-generated method stub
		
	}
}
