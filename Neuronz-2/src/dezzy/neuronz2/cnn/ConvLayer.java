package dezzy.neuronz2.cnn;

import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.utility.MatrixCondenser;

public class ConvLayer {
	final Tensor3 kernels;
	final FuncDerivPair activationFunction;
	final MatrixCondenser poolingOperator;
	double bias = Math.random() + 1;
	
	public ConvLayer(final Tensor3 _kernels, final FuncDerivPair _activationFunction, final MatrixCondenser _poolingOperator, double _bias) {
		kernels = _kernels;
		activationFunction = _activationFunction;
		poolingOperator = _poolingOperator;
		bias = _bias;
	}
	
	public Tensor3 activations(final Tensor3 prevActivations) {
		final Matrix[] matrices = new Matrix[prevActivations.dimension * kernels.dimension];
		
		return null;
	}
}
