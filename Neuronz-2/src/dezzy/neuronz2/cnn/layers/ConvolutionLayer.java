package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Tensor4;
import dezzy.neuronz2.math.utility.DoubleApplier;

public class ConvolutionLayer implements Layer {
	
	private Tensor4 filters;
	
	private Tensor3 latestInput;
	
	private static final DoubleApplier NON_MODIFIER = d -> d;

	@Override
	public Tensor3 activations(final Tensor3 prevActivations) {
		latestInput = prevActivations;
		
		final Matrix[] output = new Matrix[filters.dimension];
		
		for (int i = 0; i < filters.dimension; i++) {
			final Tensor3 kernel = filters.getTensor(i);
			final Tensor3 convolved = prevActivations.convolve(kernel, 1, NON_MODIFIER);
			output[i] = convolved.getLayer(i);
		}
		
		return new Tensor3(output);
	}

	@Override
	public Tensor3 backprop(final Tensor3 errorOutputDeriv, final double learningRate) {
		final Tensor3 
	}
	
}
