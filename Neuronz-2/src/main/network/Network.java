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
	public Tensor3 weightTensor;
	
	/**
	 * The activation vectors for the latest run of the network.
	 */
	private final Vector[] activations;
	
	/**
	 * The total number of neuron layers, including the input and output layers
	 */
	private final int layers;
	
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
			double[][] weights = new double[layerSizes[layer + 1]][layerSizes[layer] + 1]; //layerSizes[layer] + 1 to include a column for biases
			
			for (int row = 0; row < weights.length; row++) {
				for (int col = 0; col < weights[0].length; col++) {
					weights[row][col] = (double) random.nextGaussian();
					//weights[row][col] = (double) Math.random();
				}
			}
			
			weightMatrices[layer] = new Matrix(weights);
		}
		
		weightTensor = new Tensor3(weightMatrices);
		activations = new Vector[weightTensor.dimension + 1];
		layers = activations.length;
	}
	
	/**
	 * Creates a neural network by directly settings the weights and biases.
	 * 
	 * @param _weightTensor weight tensor containing weights and biases for each layer
	 */
	public Network(final Tensor3 _weightTensor) {
		weightTensor = _weightTensor;
		activations = new Vector[weightTensor.dimension + 1];
		layers = activations.length;
	}
	
	/**
	 * The partial derivative of the MSE with respect to the final output
	 */
	private static final FloatOperator MSE_DERIV = (actual, ideal) -> actual - ideal;
	
	/**
	 * Runs the network on the given inputs and returns the output.
	 * 
	 * @param input first activation vector
	 * @return output vector
	 */
	public final Vector run(final Vector input) {
		activations[0] = input;
		
		for (int i = 1; i < layers; i++) {
			if (i == layers - 1) {
				activations[i] = NetworkFunctions.computeOutputVector(weightTensor.getLayer(i - 1), activations[i - 1], Functions::sigmoid);
			} else {
				activations[i] = NetworkFunctions.computeOutputVector(weightTensor.getLayer(i - 1), activations[i - 1], Functions::sigmoid).append(1);
			}
		}
		
		return activations[layers - 1];
	}
	
	/**
	 * Apply the weight gradients with the given learning rate.
	 * 
	 * @param weightDeltas weight gradient tensor
	 * @param learningRate learning rate
	 */
	public final void applyWeightDeltas(final Tensor3 weightDeltas, final double learningRate) {
		weightTensor = weightTensor.elementOperation(weightDeltas, (w, dw) -> w - dw * learningRate);
	}
	
	/**
	 * Get the latest output from the last run of this network.
	 * 
	 * @return the latest output
	 */
	public final Vector getLatestOutput() {
		return activations[layers - 1];
	}
	
	/**
	 * Calculates and returns the weight gradients with MSE as the cost function.
	 * 
	 * @param learningRate learning rate
	 * @param input input vector (first activation vector)
	 * @param ideal expected outputs
	 * @return the weight deltas
	 */
	public final Tensor3 backprop(final double learningRate, final Vector input, final Vector ideal) {		
		run(input);
		
		final Matrix[] weightDeltaTensor = new Matrix[weightTensor.dimension]; 
		
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
			
			weightDeltaTensor[i - 1] = weightDeltas;
			
			final Matrix currentWeights = weightTensor.getLayer(i - 1);
			
			if (i == layers - 1) {
				errorOutputDeriv = currentWeights.transpose().multiply(errorInputDeriv);
			} else {
				errorOutputDeriv = currentWeights.transpose().multiply(errorInputDeriv.removeLastElement());
			}
		}
		
		return new Tensor3(weightDeltaTensor);
	}
}
