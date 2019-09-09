package main.network;

import java.util.Random;

import main.math.constructs.FuncDerivPair;
import main.math.constructs.Matrix;
import main.math.constructs.Tensor3;
import main.math.constructs.Vector;
import main.math.utility.FloatOperator;
import main.math.utility.Functions;

/**
 * A neural network. The weights and biases of the network are represented as a rank 3 tensor.
 *
 * @author Joe Desmond
 */
public final class Network {
	/**
	 * The weight relationships between each neuron, as well as biases which are stored in the last column of each matrix.
	 */
	public final Tensor3 weightTensor;
	
	/**
	 * Creates a neural network with the specified layer sizes. The weights and biases are initialized to random Standard Normal values.
	 * 
	 * @param layerSizes number of neurons in each layer, starting with the input layer and ending with the output layer, with any number of hidden layers in between
	 */
	public Network(final int ... layerSizes) {
		if (layerSizes.length < 2) {
			throw new IllegalArgumentException("Neural network must have 2 or more layers!");
		}
		
		final Matrix[] weightMatrices = new Matrix[layerSizes.length - 1];
		
		final Random random = new Random();
		for (int layer = 0; layer < layerSizes.length - 1; layer++) {
			float[][] weights = new float[layerSizes[layer + 1]][layerSizes[layer] + 1]; //layerSizes[layer] + 1 to include a column for biases
			
			for (int row = 0; row < weights.length; row++) {
				for (int col = 0; col < weights[0].length; col++) {
					weights[row][col] = (float) random.nextGaussian();
				}
			}
			
			weightMatrices[layer] = new Matrix(weights);
		}
		
		weightTensor = new Tensor3(weightMatrices);
	}
	
	/**
	 * Creates a neural network by directly settings the weights and biases.
	 * 
	 * @param _weightTensor weight tensor containing weights and biases for each layer
	 */
	public Network(final Tensor3 _weightTensor) {
		weightTensor = _weightTensor;
	}
	
	/**
	 * The partial derivative of the MSE with respect to the final output
	 */
	private static final FloatOperator MSE_DERIV = (actual, ideal) -> actual - ideal;
	
	/**
	 * Trains the network on a given set of inputs with expected outputs.
	 * 
	 * @param learningRate rate to train the network at
	 * @param input input vector (first activation vector)
	 * @param ideal expected outputs
	 * @return the actual output
	 */
	public final Vector learn(final float learningRate, final Vector input, final Vector ideal) {
		final Vector[] activations = new Vector[weightTensor.dimension + 1];
		final int layers = activations.length;
		activations[0] = input;
		
		for (int i = 1; i < layers; i++) {			
			if (i == layers - 1) {
				activations[i] = NetworkFunctions.computeOutputVector(weightTensor.getLayer(i - 1), activations[i - 1], Functions::sigmoid);
			} else {
				activations[i] = NetworkFunctions.computeOutputVector(weightTensor.getLayer(i - 1), activations[i - 1], Functions::sigmoid).append(1);
			}
		}
		
		Vector errorOutputDeriv = activations[layers - 1].elementOperation(ideal, MSE_DERIV);
		
		for (int i = layers - 1; i >= 1; i--) {
			final Vector sigmoidDeriv = activations[i].transform(FuncDerivPair.SIGMOID.partialDerivative);
			final Vector errorInputDeriv = errorOutputDeriv.hadamard(sigmoidDeriv);
			final Matrix weightDeltas;
			
			if (i == layers - 1) {
				weightDeltas = errorInputDeriv.outerProduct(activations[i - 1]);
			} else {
				weightDeltas = errorInputDeriv.removeLastElement().outerProduct(activations[i - 1]);
			}
			
			final Matrix currentWeights = weightTensor.getLayer(i - 1);
			final Matrix newWeights = currentWeights.minus(weightDeltas.transform(w -> w * learningRate));
			weightTensor.setLayer(i - 1, newWeights);
			
			if (i == layers - 1) {
				errorOutputDeriv = currentWeights.transpose().multiply(errorInputDeriv);
			} else {
				errorOutputDeriv = currentWeights.transpose().multiply(errorInputDeriv.removeLastElement());
			}
		}
		
		return activations[layers - 1];
	}
}
