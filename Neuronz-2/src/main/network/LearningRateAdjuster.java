package main.network;

/**
 * A function that adjusts the learning rate of a neural network given information about the previous epoch.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface LearningRateAdjuster {
	
	/**
	 * Returns a new learning rate given the results of the previous epoch and the current learning rate.
	 * 
	 * @param currentLearningRate current learning rate
	 * @param epoch next epoch
	 * @param prevSuccessRate previous success rate (normalized, 1.0 is 100% success)
	 * @return a learning rate to use for the next epoch
	 */
	double getNewLearningRate(double currentLearningRate, int epoch, double prevSuccessRate);
}
