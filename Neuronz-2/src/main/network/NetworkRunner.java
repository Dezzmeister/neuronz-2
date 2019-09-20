package main.network;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import main.math.constructs.Tensor3;
import main.math.constructs.Vector;
import main.math.utility.OutputVerificationScheme;
import main.network.Network.BackpropPair;

/**
 * Trains a neural network.
 *
 * @author Joe Desmond
 */
public final class NetworkRunner {
	private final Network network;
	private final InputOutputPair[] trainingData;
	private final InputOutputPair[] testData;
	private final boolean learnFromTestData;
	
	/**
	 * Creates a {@link NetworkRunner} with the given parameters. 
	 * 
	 * @param _network network to train
	 * @param trainingInputs training dataset input vectors
	 * @param trainingOutputs training dataset expected output vectors
	 * @param testInputs test dataset input vectors
	 * @param testOutputs test dataset expected output vectors
	 * @param _learnFromTestData true if the network should perform backpropagation and update weights for test data
	 */
	public NetworkRunner(final Network _network, final Vector[] trainingInputs, final Vector[] trainingOutputs, final Vector[] testInputs, final Vector[] testOutputs, final boolean _learnFromTestData) {
		network = _network;
		
		if ((trainingInputs.length != trainingOutputs.length) || (testInputs.length != testOutputs.length)) {
			throw new IllegalArgumentException("Input and output arrays must be the same length!");
		}
		
		trainingData = new InputOutputPair[trainingInputs.length];
		for (int i = 0; i < trainingInputs.length; i++) {
			trainingData[i] = new InputOutputPair(trainingInputs[i], trainingOutputs[i]);
		}
		
		testData = new InputOutputPair[testInputs.length];
		for (int i = 0; i < testInputs.length; i++) {
			testData[i] = new InputOutputPair(testInputs[i], testOutputs[i]);
		}
		
		learnFromTestData = _learnFromTestData;
	}
	
	/**
	 * Runs and trains the network, using mini-batch gradient descent. At the beginning of each epoch, the training data is shuffled. After each minibatch, the weight gradients for all entries in the minibatch are averaged
	 * and the network's weights are updated. <p>
	 * After each epoch, the network will be evaluated and {@link LearningRateAdjuster#getNewLearningRate(double, int, double) learningRateSchedule.getNewLearningRate()} will be called
	 * to adjust the learning rate for the next epoch. The initial learning rate will be determined by a call to <code>learningRateSchedule.getNewLearningRate()</code> with all parameters set to 0. <br>
	 * During evaluation, the network is run on each entry in the test dataset. {@link OutputVerificationScheme#isSuccess(Vector, Vector) successEvaluator.isSuccess()} is used to determine if the
	 * network output an appropriate vector. <br>
	 * If this {@link NetworkRunner} was created with <code>_learnFromTestData</code> equal to <code>true</code>, backpropagation will be performed and the network's weights will be updated for each test item.
	 * 
	 * @param epochs number of epochs to train for
	 * @param miniBatchSize number of entries in one minibatch
	 * @param learningRateSchedule determines the learning rate after each epoch
	 * @param successEvaluator determines if a forward pass through the network produced the expected vector (or something that is close enough to the expected vector)
	 * @param bestNetworkFileName name of file to save best network as, or null if the network should not be saved
	 * @param processingScheme what should be used to train the network
	 * @throws ExecutionException if there is a problem performing a pass through the network
	 * @throws InterruptedException if something stops one of the threads used to train the network 
	 */
	public final void run(final int epochs, final int miniBatchSize, final LearningRateAdjuster learningRateSchedule, final OutputVerificationScheme successEvaluator, final String bestNetworkFileName, final ProcessingScheme processingScheme) throws InterruptedException, ExecutionException {
		final ExecutorService threadPool = (processingScheme == ProcessingScheme.CPU_MULTITHREADED) ? Executors.newFixedThreadPool(miniBatchSize) : null;
		
		double highestSuccessRate = 0;
		double learningRate = learningRateSchedule.getNewLearningRate(0, 0, 0);
		
		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.println("Epoch " + epoch + ":");
			System.out.println("\tLearning Rate: " + learningRate);
			long time = System.currentTimeMillis();
			double previousSuccessRate = 0;
			Collections.shuffle(Arrays.asList(trainingData));
			
			Tensor3 weightDeltas = network.weightTensor.transform(w -> 0);
			
			for (int j = 0; j < trainingData.length; j += miniBatchSize) {
				
				switch (processingScheme) {
					case CPU_MULTITHREADED:
						@SuppressWarnings("unchecked")
						final Future<Tensor3>[] miniBatchResults = (Future<Tensor3>[]) new Future<?>[miniBatchSize];
						for (int k = 0; k < miniBatchSize; k++) {
							final NetworkPass networkPass = new NetworkPass(network, trainingData[k + j].input, trainingData[k + j].output);
							final Future<Tensor3> future = threadPool.submit(networkPass);
							miniBatchResults[k] = future;
						}
						
						for (int k = 0; k < miniBatchSize; k++) {
							final Tensor3 gradients = miniBatchResults[k].get();
							weightDeltas = weightDeltas.plus(gradients);
						}
						break;
					case CPU_SINGLE_THREAD:
						for (int k = 0; k < miniBatchSize; k++) {
							final Tensor3 weightGradient = network.backprop(trainingData[k + j].input, trainingData[k + j].output).weightDeltas;
							weightDeltas = weightDeltas.plus(weightGradient);
						}
						break;
					case GPU:
						break;
					case FPGA:
						break;
				}
				
				weightDeltas = weightDeltas.transform(w -> w / (double)miniBatchSize);
				network.applyWeightDeltas(weightDeltas, learningRate);
				weightDeltas = weightDeltas.transform(w -> 0);
			}
			
			int successes = 0;
			for (int j = 0; j < testData.length; j++) {
				final Vector[] activations;
				
				if (learnFromTestData) {
					final BackpropPair result = network.backprop(testData[j].input, testData[j].output);
					network.applyWeightDeltas(result.weightDeltas, learningRate);
					activations = result.activations;
				} else {
					activations = network.run(testData[j].input);
				}
				
				if (successEvaluator.isSuccess(network.getLatestOutput(activations), testData[j].output)) {
					successes++;
				}
			}
			
			previousSuccessRate = successes/(double)testData.length;
			learningRate = learningRateSchedule.getNewLearningRate(learningRate, epoch + 1, previousSuccessRate);
			
			if (previousSuccessRate > highestSuccessRate && bestNetworkFileName != null) {
				highestSuccessRate = previousSuccessRate;
				
				try {
					network.saveAs(bestNetworkFileName);
				} catch (IOException e) {
					System.err.println("Error saving best network run!");
					e.printStackTrace();
				}
			}
			double timeInSeconds = (System.currentTimeMillis() - time) / (double)1000;
			System.out.println("\tSuccess rate: " + (100.0 * previousSuccessRate) + "%");
			System.out.println("\tCompleted in " + timeInSeconds + " seconds");
		}
		
		threadPool.shutdown();
	}
	
	/**
	 * An input vector and its associated expected output vector.
	 *
	 * @author Joe Desmond
	 */
	private static final class InputOutputPair {
		
		/**
		 * Input vector
		 */
		final Vector input;
		
		/**
		 * Expected output vector
		 */
		final Vector output;
		
		/**
		 * Creates an {@link InputOutputPair} with the given input and expected output vectors.
		 * 
		 * @param _input input vector
		 * @param _output expected output vector
		 */
		InputOutputPair(final Vector _input, final Vector _output) {
			input = _input;
			output = _output;
		}
	}
}
