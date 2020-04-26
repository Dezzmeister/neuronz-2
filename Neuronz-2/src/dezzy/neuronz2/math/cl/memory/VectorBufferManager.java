package dezzy.neuronz2.math.cl.memory;


public class VectorBufferManager {
	
	private final MemoryManager memoryManager;
	
	private final int numBuffers;
	
	private final boolean[] inUse;
	
	public VectorBufferManager(final MemoryManager _memoryManager, final int _numBuffers) {
		memoryManager = _memoryManager;
		numBuffers = _numBuffers;
		inUse = new boolean[numBuffers];
	}
}
