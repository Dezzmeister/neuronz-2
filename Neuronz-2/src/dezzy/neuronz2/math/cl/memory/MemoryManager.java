package dezzy.neuronz2.math.cl.memory;

/**
 * Defines two native methods for aligned memory allocation. These methods allow memory to be shared between the CPU and GPU
 * on Intel systems (zero-copy memory), which increases performance by requiring less buffer copies.
 *
 * @author Joe Desmond
 */
public class MemoryManager {
	
	static {
		
		// Load the native functions
		System.loadLibrary("src/dezzy/neuronz2/math/cl/memory/memory_manager");
	}
	
	/**
	 * Calls <code>_aligned_malloc(size_t size, size_t alignment)</code> from
	 * <code>stdlib.h</code>. Storage is <b>NOT</b> initialized.
	 * <p>
	 * <b>IMPORTANT: MEMORY MUST BE FREED WITH {@link #alignedFree(long)}</b>
	 * 
	 * @param size number of <code>doubles</code> to allocate (must be an integer multiple of <code>alignment</code>)
	 * @param alignment memory alignment (must be a power of two)
	 * @return a pointer to the newly allocated memory
	 */
	public native double[] alignedMalloc(long size, long alignment);
	
	/**
	 * Frees memory allocated with {@link #alignedMalloc(long, long)}.
	 * 
	 * @param ptr a pointer to the memory
	 */
	public native void alignedFree(double[] ptr);
}
