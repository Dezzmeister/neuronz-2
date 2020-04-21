package dezzy.neuronz2.cnn;

import java.util.ArrayList;
import java.util.List;

import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.cnn.pooling.PoolingResult;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Tensor4;

public class ConvLayer {
	final Tensor4 filters;
	final FuncDerivPair activationFunction;
	final PoolingOperation poolingOperator;
	final int poolRows;
	final int poolCols;
	double bias = Math.random() + 1;
	
	Tensor3 modifiedInput;
	Tensor3 convolved;
	
	public ConvLayer(final Tensor4 _filters, final FuncDerivPair _activationFunction, final PoolingOperation _poolingOperator, final int _poolRows, final int _poolCols, double _bias) {
		filters = _filters;
		activationFunction = _activationFunction;
		poolingOperator = _poolingOperator;
		poolRows = _poolRows;
		poolCols = _poolCols;
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
		
		convolved = new Tensor3(matrices.toArray(new Matrix[matrices.size()]));
		
		final Matrix[] pooledMatrices = new Matrix[matrices.size()];
		final Matrix[] modifiedInputMatrices = new Matrix[matrices.size()];
		
		for (int i = 0; i < matrices.size(); i++) {
			final PoolingResult result = matrices.get(i).poolingTransform(poolRows, poolCols, poolRows, poolCols, poolingOperator);
			pooledMatrices[i] = result.result;
			modifiedInputMatrices[i] = result.modifiedInput;
		}
		
		modifiedInput = new Tensor3(modifiedInputMatrices);
		
		final Tensor3 pooled = new Tensor3(pooledMatrices);
		
		final Tensor3 activations = pooled.transform(activationFunction.function);
		
		return activations;
	}
	
	public void backprop(final Tensor3 errorOutputDeriv) {
		for (int m = 0; m < errorOutputDeriv.dimension; m++) {
			final Matrix current
		}
	}
}
