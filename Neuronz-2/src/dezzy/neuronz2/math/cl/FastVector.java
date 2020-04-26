package dezzy.neuronz2.math.cl;

import java.io.IOException;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import dezzy.neuronz2.cl.CLState;
import dezzy.neuronz2.cl.CLUtilities;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.DimensionMismatchException;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Implements vector operations with OpenCL kernels instead of Java code. This allows them to be parallelized and 
 * performed on the graphics card.
 * <p>
 * <b>NOTE:</b> The optimized functions in this class need to be initialized with 
 * {@link #initialize(cl_context, cl_command_queue)}. Similarly, when this class is no longer needed, they should be
 * released with {@link #release()}.
 *
 * @author Joe Desmond
 */
public class FastVector extends Vector {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -3850100418055018748L;
	
	/**
	 * OpenCL context
	 */
	private static cl_context context;
	
	/**
	 * OpenCL command queue
	 */
	private static cl_command_queue commandQueue;
	
	/**
	 * OpenCL program containing all functions used by this class
	 */
	private static cl_program program;
	
	/**
	 * Initializes the FastVector class and loads OpenCL kernels for the optimized functions. Kernel code is
	 * stored at <code>"kernels/math/vector"</code>, and the kernels are compiled in this function.
	 * 
	 * @param clState OpenCL state object
	 * @throws IOException if there is a problem loading the kernels
	 */
	public static void initialize(final CLState clState) throws IOException {
		context = clState.context;
		commandQueue = clState.commandQueue;
		
		final String hadamardCode = CLUtilities.readString("kernels/math/vector/hadamard.c");
		
		final String[] programStrings = {hadamardCode};
		
		program = CL.clCreateProgramWithSource(context, programStrings.length, programStrings, null, null);
		CL.clBuildProgram(program, 0, null, null, null, null);
        
        hadamardKernel = CL.clCreateKernel(program, "vectorHadamard", null);
	}
	
	/**
	 * Releases the functions/kernels used by this class.
	 */
	public static void release() {
		CL.clReleaseKernel(hadamardKernel);
		
		CL.clReleaseProgram(program);
	}
	
	/**
	 * Creates a FastVector with the given component values.
	 * 
	 * @param _components the components of this vector
	 */
	public FastVector(final double ... _components) {
		super(_components);
	}
	
	/**
	 * Generates a fast vector with the specified length using the given generator function. Calls
	 * {@link IndexedGenerator#generate(int...) generator.generate(i)} with every component index
	 * to obtain the values of the vector.
	 * 
	 * @param generator generator function, used to generate components of the vector
	 * @param length number of components in the vector
	 * @return a new FastVector
	 */
	public static FastVector generate(final IndexedGenerator generator, final int length) {
		final double[] out = new double[length];
		
		for (int i = 0; i < length; i++) {
			out[i] = generator.generate(i);
		}
		
		return new FastVector(out);
	}
	
	/**
	 * Hadamard product kernel
	 */
	private static cl_kernel hadamardKernel;
	
	
	@Override
	public Vector hadamard(final Vector other) {
		if (dimension != other.dimension) {
			throw new DimensionMismatchException("Vectors must have same dimensions to compute hadamard product!");
		}
		
		final double[] otherComponents = getComponents(other);
		final double[] result = new double[dimension];
		
		final Pointer srcA = Pointer.to(components);
		final Pointer srcB = Pointer.to(otherComponents);
		final Pointer dst = Pointer.to(result);
		
		final cl_mem srcMemA = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * dimension, srcA, null);
        final cl_mem srcMemB = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * dimension, srcB, null);
        final cl_mem dstMem = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_double * dimension, null, null);
        
        
        CL.clSetKernelArg(hadamardKernel, 0, Sizeof.cl_mem, Pointer.to(srcMemA));
        CL.clSetKernelArg(hadamardKernel, 1, Sizeof.cl_mem, Pointer.to(srcMemB));
        CL.clSetKernelArg(hadamardKernel, 2, Sizeof.cl_mem, Pointer.to(dstMem));
        
        final long[] global_work_size = {dimension};
        
        CL.clEnqueueNDRangeKernel(commandQueue, hadamardKernel, 1, null, global_work_size, null, 0, null, null);
        
        CL.clEnqueueReadBuffer(commandQueue, dstMem, CL.CL_TRUE, 0, dimension * Sizeof.cl_double, dst, 0, null, null);
        
        CL.clReleaseMemObject(srcMemA);
        CL.clReleaseMemObject(srcMemB);
        CL.clReleaseMemObject(dstMem);
        
        return new FastVector(result);
	}
}
