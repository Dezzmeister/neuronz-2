package dezzy.neuronz2.cl;

import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;
import org.jocl.cl_queue_properties;

/**
 * Contains various OpenCL state objects. These objects will generally persist while Neuronz-2
 * is in use. The objects contained here are used in fast math classes such as 
 * {@link dezzy.neuronz2.math.cl.FastVector FastVector}. An OpenCL state can be created and initialized
 * with {@link #createInstance(long)}.
 * <p>
 * <b>When this object is no longer needed, {@link #release()} should be called.</b>
 *
 * @author Joe Desmond
 */
public class CLState {
	
	/**
	 * OpenCL platform ID
	 */
	public final cl_platform_id platform;
	
	/**
	 * OpenCL device ID
	 */
	public final cl_device_id device;
	
	/**
	 * OpenCL context
	 */
	public final cl_context context;
	
	/**
	 * OpenCL command queue
	 */
	public final cl_command_queue commandQueue;
	
	/**
	 * Creates a CLState with the given OpenCL state objects.
	 * 
	 * @param _platform OpenCL platform ID
	 * @param _device OpenCL device ID
	 * @param _context OpenCL context
	 * @param _commandQueue OpenCL command queue
	 */
	public CLState(final cl_platform_id _platform, final cl_device_id _device, final cl_context _context, final cl_command_queue _commandQueue) {
		platform = _platform;
		device = _device;
		context = _context;
		commandQueue = _commandQueue;
	}
	
	/**
	 * Creates and initializes a CLState with the given device type.
	 * 
	 * @param deviceType desired device type: {@link CL#CL_DEVICE_TYPE_ALL ALL}, {@link CL#CL_DEVICE_TYPE_GPU GPU}, etc.
	 * @return a new CLState
	 */
	public static final CLState createInstance(final long deviceType) {
		final int platformIndex = 0;
        final int deviceIndex = 0;
		
		CL.setExceptionsEnabled(true);
		
		final int[] numPlatformsArray = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		final int numPlatforms = numPlatformsArray[0];
		
		
		final cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(platforms.length, platforms, null);
        final cl_platform_id platform = platforms[platformIndex];
        
        
        final cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
        
        
        final int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        final int numDevices = numDevicesArray[0];
        
        
        final cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        final cl_device_id device = devices[deviceIndex];
        
        
        final cl_context context = CL.clCreateContext(contextProperties, 1, new cl_device_id[] {device}, null, null, null);
        
        final cl_queue_properties properties = new cl_queue_properties();
        final cl_command_queue commandQueue = CL.clCreateCommandQueueWithProperties(context, device, properties, null);
        
        return new CLState(platform, device, context, commandQueue);
	}
	
	/**
	 * Releases the OpenCL context and command queue associated with this object.
	 */
	public void release() {
		CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);
	}
}
