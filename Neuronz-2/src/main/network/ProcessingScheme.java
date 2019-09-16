package main.network;

/**
 * How {@link NetworkRunner} should train a network.
 *
 * @author Joe Desmond
 */
public enum ProcessingScheme {
	
	/**
	 * Use one thread for everything
	 */
	CPU_SINGLE_THREAD,
	
	/**
	 * Split the job across multiple threads
	 */
	CPU_MULTITHREADED,
	
	/**
	 * Use the graphics card
	 */
	GPU
}