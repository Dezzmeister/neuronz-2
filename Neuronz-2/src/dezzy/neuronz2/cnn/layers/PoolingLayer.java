package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.cnn.pooling.PoolingResult;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * A pooling layer in a convolutional neural network.
 *
 * @author Joe Desmond
 */
public class PoolingLayer implements Layer {
	
	/**
	 * The pooling operation
	 */
	private final PoolingOperation poolingOperation;
	
	/**
	 * The number of rows in the pooling window (also used as the row stride)
	 */
	private final int windowRows;
	
	/**
	 * The number of columns in the pooling window (also used as the column stride)
	 */
	private final int windowCols;
	
	/**
	 * The modified input to the pooling operation, used in backpropagation
	 */
	private Tensor3 modifiedInput;
	
	/**
	 * Constructs a pooling layer with the given pooling operation and pooling window size.
	 * 
	 * @param _poolingOperation pooling operation (Example: {@link PoolingOperation#MAX_POOLING})
	 * @param _windowRows number of rows in the pooling window
	 * @param _windowCols number of columns in the pooling window
	 */
	public PoolingLayer(final PoolingOperation _poolingOperation, final int _windowRows, final int _windowCols) {
		poolingOperation = _poolingOperation;
		windowRows = _windowRows;
		windowCols = _windowCols;
	}
	
	/**
	 * Applies {@linkplain #poolingOperation this} pooling operation layer-wise to the given activations and returns a smaller tensor
	 * (same number of layers; but smaller layers) for the next network layer. Saves a {@linkplain PoolingResult#modifiedInput modified version} 
	 * of the input tensor to be used for backpropagation.
	 * 
	 * @param prevActivations output of previous network layer
	 * @return output of this network layer
	 */
	@Override
	public Tensor3 activations(final Tensor3 prevActivations) {
		final Matrix[] output = new Matrix[prevActivations.dimension];
		final Matrix[] newInputs = new Matrix[prevActivations.dimension];
		
		for (int i = 0; i < prevActivations.dimension; i++) {
			final PoolingResult result = prevActivations.getLayer(i).poolingTransform(windowRows, windowCols, windowRows, windowCols, poolingOperation);
			
			output[i] = result.result;
			newInputs[i] = result.modifiedInput;
		}
		
		modifiedInput = new Tensor3(newInputs);
		
		return new Tensor3(output);
	}
	
	/**
	 * Expands <code>errorOutputDeriv</code> so that it has the same shape as the 
	 * {@linkplain PoolingResult#modifiedInput modified input} to this layer, 
	 * then multiplies the expanded tensor element-wise by the modified input (chain rule).
	 * 
	 * @param errorOutputDeriv the (partial) derivative of the network error with respect to the output of this layer
	 * @return the (partial) derivative of the network error with respect to the input to this layer
	 */
	@Override
	public Tensor3 backprop(final Tensor3 errorOutputDeriv) {
		final Matrix[] errorInputDeriv = new Matrix[errorOutputDeriv.dimension];
		
		for (int m = 0; m < errorOutputDeriv.dimension; m++) {
			final Matrix pooled = errorOutputDeriv.getLayer(m);
			final Matrix input = modifiedInput.getLayer(m);
			
			errorInputDeriv[m] = pooled.expandAndMultiply(input);
		}
		
		return new Tensor3(errorInputDeriv);
	}	
}
