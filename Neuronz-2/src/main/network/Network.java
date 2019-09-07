package main.network;

import java.util.Random;

import main.math.constructs.Matrix;
import main.math.constructs.Tensor3;

/**
 * A neural network.
 *
 * @author Joe Desmond
 */
public final class Network {
	/**
	 * The weight relationships between each neuron, as well as biases represented as the last column of each matrix.
	 */
	private final Tensor3 weightTensor;
	
	/**
	 * Creates a neural network with the specified layer sizes. The weights and biases are initialized to random normalized values.
	 * 
	 * @param layerSizes
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
}
