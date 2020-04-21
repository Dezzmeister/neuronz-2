package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.cnn.pooling.PoolingResult;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;

public class PoolingLayer implements Layer {
	
	private final PoolingOperation poolingOperation;
	private final int windowRows;
	private final int windowCols;
	
	private Tensor3 modifiedInput;
	
	public PoolingLayer(final PoolingOperation _poolingOperation, final int _windowRows, final int _windowCols) {
		poolingOperation = _poolingOperation;
		windowRows = _windowRows;
		windowCols = _windowCols;
	}

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
