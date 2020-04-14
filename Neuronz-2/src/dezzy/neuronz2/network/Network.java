package dezzy.neuronz2.network;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.DoubleApplier;
import dezzy.neuronz2.math.utility.DoubleOperator;


/**
 * A neural network. The weights and biases of the network are represented as a rank 3 tensor.
 *
 * @author Joe Desmond
 */
public final class Network implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4591608638507838712L;

	/**
	 * The weight relationships between each neuron, as well as biases which are stored in the last column of each matrix.
	 */
	public Tensor3 weightTensor;
	
	/**
	 * Derivative of the cost function to use for backpropagation
	 */
	private final DoubleOperator costFunctionDerivative;
	
	/**
	 * Activation functions for each layer, except the output layer
	 */
	private final FuncDerivPair[] activationFunctions;
	
	/**
	 * The total number of neuron layers, including the input and output layers
	 */
	private final int layers;
	
	/**
	 * Creates a neural network with the specified layer sizes. The weights and biases are initialized to random Standard Normal values. A network created with this
	 * constructor will use the sigmoid activation function for each layer and MSE as the cost function. To change these functions, use {@link Network#Network(int[], FuncDerivPair[], DoubleOperator)}.
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
				}
			}
			
			weightMatrices[layer] = new Matrix(weights);
		}
		
		weightTensor = new Tensor3(weightMatrices);
		layers = weightTensor.dimension + 1;
		
		activationFunctions = new FuncDerivPair[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			activationFunctions[i] = FuncDerivPair.SIGMOID;
		}
		
		costFunctionDerivative = MSE_DERIV;
	}
	
	/**
	 * Same as {@link Network#Network(Tensor3)}, except the cost function can be set and activation functions can be set for each layer (except output).
	 * 
	 * @param layerSizes number of neurons in each layer, including input and output layers
	 * @param _activationFunctions activation functions for each layer (except output layer)
	 * @param _costFunctionDerivative derivative of the cost function to use for backpropagation: first operand is the actual output, second is the ideal output
	 */
	public Network(final int[] layerSizes, final FuncDerivPair[] _activationFunctions, final DoubleOperator _costFunctionDerivative) {
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
				}
			}
			
			weightMatrices[layer] = new Matrix(weights);
		}
		
		weightTensor = new Tensor3(weightMatrices);
		layers = weightTensor.dimension + 1;
		
		activationFunctions = _activationFunctions;
		costFunctionDerivative = _costFunctionDerivative;
	}
	
	/**
	 * Creates a neural network by directly settings the weights and biases.
	 * 
	 * @param _weightTensor weight tensor containing weights and biases for each layer
	 */
	public Network(final Tensor3 _weightTensor) {
		weightTensor = _weightTensor;
		layers = weightTensor.dimension + 1;
		
		activationFunctions = new FuncDerivPair[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			activationFunctions[i] = FuncDerivPair.SIGMOID;
		}
		
		costFunctionDerivative = MSE_DERIV;
	}
	
	
	
	/**
	 * The partial derivative of the MSE with respect to the final output
	 */
	public static final DoubleOperator MSE_DERIV = (actual, ideal) -> actual - ideal;
	
	/**
	 * Runs the network on the given inputs and returns the activations.
	 * 
	 * @param input first activation vector
	 * @return output vector
	 */
	public final Vector[] run(final Vector input) {
		final Vector[] activations = new Vector[weightTensor.dimension + 1];
		activations[0] = input.append(1);
		
		for (int i = 1; i < layers; i++) {
			final DoubleApplier activationFunction = activationFunctions[i - 1].function;
			
			if (i == layers - 1) {
				activations[i] = NetworkFunctions.computeOutputVector(weightTensor.getLayer(i - 1), activations[i - 1], activationFunction);
			} else {
				activations[i] = NetworkFunctions.computeOutputVector(weightTensor.getLayer(i - 1), activations[i - 1], activationFunction).append(1);
			}
		}
		
		return activations;
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
	 * @param activations activation vector from the latest run
	 * @return the latest output
	 */
	public final Vector getLatestOutput(final Vector[] activations) {
		return activations[layers - 1];
	}
	
	/**
	 * Calculates and returns the weight gradients. Uses matrix operations instead of explicit loops.
	 * 
	 * @param input input vector (first activation vector)
	 * @param ideal expected outputs
	 * @return the weight deltas and activations
	 */
	public final BackpropTriplet backprop(final Vector input, final Vector ideal) {		
		final Vector[] activations = run(input);
		
		final Matrix[] weightDeltaTensor = new Matrix[weightTensor.dimension]; 
		
		Vector errorOutputDeriv = activations[layers - 1].elementOperation(ideal, costFunctionDerivative);	//dE/dout
		
		for (int i = layers - 1; i >= 1; i--) {
			final DoubleApplier activationFuncDeriv = activationFunctions[i - 1].derivative;
			
			final Vector activationRawInputDeriv = activations[i].transform(activationFuncDeriv);	//dout/din
			final Vector errorInputDeriv = errorOutputDeriv.hadamard(activationRawInputDeriv);	//dE/din
			final Matrix weightDeltas;	//dE/dW
			
			if (i == layers - 1) {
				weightDeltas = errorInputDeriv.outerProduct(activations[i - 1]);	//Rank 2 Tensor!
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
		
		return new BackpropTriplet(new Tensor3(weightDeltaTensor), activations, errorOutputDeriv);
	}
	
	/**
	 * Saves this network to a file so that it can be run/trained later.
	 * 
	 * @param path path to file (will be created if it doesn't exist)
	 * @throws IOException if there is a problem creating the {@link FileOutputStream} or {@link ObjectOutputStream}
	 */
	public final void saveAs(final String path) throws IOException {
		final FileOutputStream fos = new FileOutputStream(path);
		final ObjectOutputStream oos = new ObjectOutputStream(fos);
		
		oos.writeObject(this);
		oos.close();
	}
	
	/**
	 * Loads a network from a file. Networks can be saved to a file by {@link Network#saveAs}.
	 * 
	 * @param path path to network file
	 * @return a network loaded from <code>path</code>
	 * @throws IOException if there is a problem creating the {@link FileInputStream} or {@link ObjectInputStream}
	 * @throws ClassNotFoundException if the file at <code>path</code> does not contain a {@link Network}
	 */
	public static final Network loadFrom(final String path) throws IOException, ClassNotFoundException {
		final FileInputStream fis = new FileInputStream(path);
		final ObjectInputStream ois = new ObjectInputStream(fis);
		final Network network = (Network) ois.readObject();
		
		ois.close();
		return network;
	}
	
	/**
	 * The result of one forward pass and one backward pass through the network, contains weight deltas and the activations for each layer.
	 *
	 * @author Joe Desmond
	 */
	public static final class BackpropTriplet {
		/**
		 * Weight gradient
		 */
		public final Tensor3 weightDeltas;
		
		/**
		 * Activations for each layer
		 */
		public final Vector[] activations;
		
		/**
		 * Derivative of the error with respect to the original inputs to the network
		 */
		public final Vector errorInputDeriv;
		
		/**
		 * Creates a BackpropPair representing the result of one forward and backward pass through the network.
		 * 
		 * @param _weightDeltas weight gradient from backward pass
		 * @param _activations activations from forward pass
		 * @param _errorInputDeriv derivative of the error with respect to the original network inputs
		 */
		public BackpropTriplet(final Tensor3 _weightDeltas, final Vector[] _activations, final Vector _errorInputDeriv) {
			weightDeltas = _weightDeltas;
			activations = _activations;
			errorInputDeriv = _errorInputDeriv;
		}
	}
}
