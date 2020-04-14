package dezzy.neuronz2.cnn;

import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.utility.MatrixCondenser;

public class ConvLayer {
	final Matrix[] kernels;
	final FuncDerivPair activationFunction;
	final MatrixCondenser poolingOperator;
	double bias = Math.random() + 1;
	
	public ConvLayer(final Matrix[] _kernels, final FuncDerivPair _activationFunction, final MatrixCondenser _poolingOperator, double _bias) {
		kernels = _kernels;
		activationFunction = _activationFunction;
		poolingOperator = _poolingOperator;
		bias = _bias;
	}
}
