package main.network;

import java.util.Random;

import main.math.constructs.Matrix;
import main.math.constructs.Tensor3;
import main.math.constructs.Vector;

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
	
	public final Vector feedForward(final Vector input) {
		//TODO: Implement this
		return null;
	}
	
	/**
	 * Creates a neural network by directly settings the weights and biases.
	 * 
	 * @param _weightTensor weight tensor containing weights and biases for each layer
	 */
	public Network(final Tensor3 _weightTensor) {
		weightTensor = _weightTensor;
	}
}
