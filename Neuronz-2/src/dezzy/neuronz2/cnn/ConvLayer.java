package dezzy.neuronz2.cnn;

import java.util.ArrayList;
import java.util.List;

import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Tensor4;

public class ConvLayer {
	final Tensor4 filters;
	final FuncDerivPair activationFunction;
	final PoolingOperation poolingOperator;
	double bias = Math.random() + 1;
	
	public ConvLayer(final Tensor4 _filters, final FuncDerivPair _activationFunction, final PoolingOperation _poolingOperator, double _bias) {
		filters = _filters;
		activationFunction = _activationFunction;
		poolingOperator = _poolingOperator;
		bias = _bias;
	}
	
	public Tensor3 activations(final Tensor3 prevActivations) {
		final List<Matrix> matrices = new ArrayList<Matrix>();
		
		for (int i = 0; i < filters.dimension; i++) {
			final Tensor3 tensor = prevActivations.convolve(filters.getTensor(i), 1, d -> d);
			
			for (int m = 0; m < tensor.dimension; m++) {
				matrices.add(tensor.getLayer(m));
			}
		}	
		
		final Tensor3 convolved = new Tensor3((Matrix[])matrices.toArray());
		final Tensor3 modified = convolved.transform(activationFunction.function);
		
		return modified;
	}
	
	public void backprop(final Matrix nextActivations) {
		
	}
}
