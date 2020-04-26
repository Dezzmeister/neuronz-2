package dezzy.neuronz2.math.cl.test;

import java.io.IOException;
import java.util.Random;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import dezzy.neuronz2.cl.CLState;
import dezzy.neuronz2.cl.CLUtilities;
import dezzy.neuronz2.math.cl.FastVector;
import dezzy.neuronz2.math.cl.memory.MemoryManager;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Functions for testing GPU math optimizations and comparing the performance benefits.
 *
 * @author Joe Desmond
 */
@SuppressWarnings("unused")
public class FastMathTest {
	
	public static final void main(final String[] args) throws IOException {
		//simpleKernelTest();
		//frameworkFunctionTest();
		lowLevelMemoryTest();
	}
	
	private static final void lowLevelMemoryTest() {
		final MemoryManager memory = new MemoryManager();
		final double[] ptr = memory.alignedMalloc(64, 64);
		ptr[63] = 1738.16;
		
		System.out.println(ptr[63]);
		memory.alignedFree(ptr);
		System.out.println("No exceptions");
	}
	
	/**
	 * Tests some of the framework functions for OpenCL operations.
	 * 
	 * @throws IOException if there is a problem loading OpenCL kernel code
	 */
	private static final void frameworkFunctionTest() throws IOException {
		final CLState clState = CLState.createInstance(CL.CL_DEVICE_TYPE_GPU);
		FastVector.initialize(clState);
		
		final Random random = new Random();
		final int length = 100;
		
		final double[] vals0 = new double[length];
		final double[] vals1 = new double[length];
		
		for (int i = 0; i < length; i++) {
			vals0[i] = random.nextGaussian();
			vals1[i] = random.nextGaussian();
		}
		
		final Vector v0 = new Vector(vals0);
		final Vector v1 = new Vector(vals1);
		
		final Vector fv0 = new FastVector(vals0);
		final Vector fv1 = new FastVector(vals1);
		
		long nanoTime0;
		long nanoTime1;
		
		nanoTime0 = System.nanoTime();
		v0.hadamard(v1);
		nanoTime1 = System.nanoTime();
		
		System.out.println("Standard Vector: " + (nanoTime1 - nanoTime0) + " ns");
		
		nanoTime0 = System.nanoTime();
		fv0.hadamard(fv1);
		nanoTime1 = System.nanoTime();
		
		System.out.println("Fast Vector: " + (nanoTime1 - nanoTime0) + " ns");
		
		FastVector.release();
		clState.release();
	}
	
	/**
	 * Creates an OpenCL context and tests the vectorHadamard kernel.
	 * @throws IOException 
	 */
	private static final void simpleKernelTest() throws IOException {
		final Random random = new Random();
		final int length = 50;
		
		final double[] vec0 = new double[length];
		final double[] vec1 = new double[length];
		final double[] result = new double[length];
		
		for (int i = 0; i < length; i++) {
			vec0[i] = random.nextGaussian();
			vec1[i] = random.nextGaussian();
		}		
		
		final Pointer srcA = Pointer.to(vec0);
		final Pointer srcB = Pointer.to(vec1);
		final Pointer dst = Pointer.to(result);
		
		final int platformIndex = 0;
        final long deviceType = CL.CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;
		
		CL.setExceptionsEnabled(true);
		
		final int[] numPlatformsArray = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		final int numPlatforms = numPlatformsArray[0];
		
		System.out.println("Platforms: " + numPlatforms);
		
		cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];
        
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
        
        final int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        final int numDevices = numDevicesArray[0];
        System.out.println("Devices: " + numDevices);
        
        cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];
        
        
        for (int i = 0; i < numDevices; i++) {
        	final byte[] nameString = new byte[256];
        	 
        	
        	CL.clGetDeviceInfo(devices[i], CL.CL_DEVICE_NAME, 256, Pointer.to(nameString), null);
        	System.out.println(new String(nameString));
        }
        
        cl_context context = CL.clCreateContext(contextProperties, 1, new cl_device_id[] {device}, null, null, null);
        
        cl_queue_properties properties = new cl_queue_properties();
        cl_command_queue commandQueue = CL.clCreateCommandQueueWithProperties(context, device, properties, null);
        
        cl_mem srcMemA = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * length, srcA, null);
        cl_mem srcMemB = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * length, srcB, null);
        cl_mem dstMem = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_double * length, null, null);
        
        final String kernelCode = CLUtilities.readString("kernels/math/vector/hadamard.c");
        
        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[] {kernelCode}, null, null);
        
        CL.clBuildProgram(program, 0, null, null, null, null);
        
        cl_kernel kernel = CL.clCreateKernel(program, "vectorHadamard", null);
        
        int a = 0;
        CL.clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemA));
        CL.clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemB));
        CL.clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(dstMem));
        
        final long[] global_work_size = {length};
        
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, null, 0, null, null);
        
        CL.clEnqueueReadBuffer(commandQueue, dstMem, CL.CL_TRUE, 0, length * Sizeof.cl_double, dst, 0, null, null);
        
        CL.clReleaseMemObject(srcMemA);
        CL.clReleaseMemObject(srcMemB);
        CL.clReleaseMemObject(dstMem);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);
        
        for (int i = 0; i < length; i++) {
        	System.out.println(result[i]);
        }
	}
}
