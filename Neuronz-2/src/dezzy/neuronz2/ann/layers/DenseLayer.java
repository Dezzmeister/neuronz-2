package dezzy.neuronz2.ann.layers;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * A dense (fully connected) layer in a neural network.
 *
 * @author Joe Desmond
 */
public class DenseLayer implements Layer<Vector, Vector> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -289894409148356984L;
	
	/**
	 * Weight matrix
	 */
	private Matrix weights;
	
	/**
	 * Bias vector
	 */
	private Vector bias;
	
	/**
	 * Accumulated weight deltas, used in backpropagation
	 */
	private Matrix weightDeltas = null;
	
	/**
	 * Accumulated bias deltas, used in backpropagation
	 */
	private Vector biasDeltas = null;
	
	/**
	 * The latest input to this layer
	 */
	private Vector latestInput;
	
	/**
	 * Constructs a fully connected layer with the given initial weights and biases.
	 * 
	 * @param _weights weights
	 * @param _bias biases
	 */
	public DenseLayer(final Matrix _weights, final Vector _bias) {
		weights = _weights;
		bias = _bias;
	}
	
	@Override
	public Vector forwardPass(final Vector prevActivations) {
		final Vector multiplied = weights.multiply(prevActivations);
		
		latestInput = prevActivations;
		
		return multiplied.plus(bias);
	}

	@Override
	public Vector backprop(final Vector errorOutputDeriv, final boolean isFirstLayer) {
		final Matrix newWeightDeltas = errorOutputDeriv.outerProduct(latestInput);
		final Vector newBiasDeltas = errorOutputDeriv;
		
		if (weightDeltas == null) {
			weightDeltas = newWeightDeltas;
		} else {
			weightDeltas = weightDeltas.plus(newWeightDeltas);
		}
		
		if (biasDeltas == null) {
			biasDeltas = newBiasDeltas;
		} else {
			biasDeltas = biasDeltas.plus(newBiasDeltas);
		}
		
		if (isFirstLayer) {
			return null;
		}
		
		final Vector output = weights.transpose().multiply(errorOutputDeriv);
		
		return output;
	}

	@Override
	public void update(final double learningRate) {
		final Matrix weightGradient = weightDeltas.transform(w -> learningRate * w);
		final Vector biasGradient = biasDeltas.transform(b -> learningRate * b);
		
		weights = weights.minus(weightGradient);
		bias = bias.minus(biasGradient);
		
		weightDeltas = null;
		biasDeltas = null;
	}
	
	@Override
	public int parameterCount() {
		return bias.dimension + (weights.rows * weights.cols);
	}
}
