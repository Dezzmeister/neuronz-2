package dezzy.neuronz2.driver;

import static jcuda.driver.JCudaDriver.cuInit;

import jcuda.driver.JCudaDriver;

import jcuda.driver.CUresult;

/**
 * Initializes the JCuda Driver API.
 * 
 * @author Joe Desmond
 */
public class Init {
	
	/**
	 * The status code returned by {@link jcuda.driver.JCudaDriver#cuInit(int) cuInit(int)}.
	 * May be one of {@link CUresult#CUDA_SUCCESS CUDA_SUCCESS}, 
	 * {@link CUresult#CUDA_ERROR_INVALID_VALUE CUDA_ERROR_INVALID_VALUE}, or 
	 * {@link CUresult#CUDA_ERROR_INVALID_DEVICE CUDA_ERROR_INVALID_DEVICE}.
	 */
	public static final int CUDA_DRIVER_STATUS_CODE;
	
	static {
		JCudaDriver.setExceptionsEnabled(true);
		CUDA_DRIVER_STATUS_CODE = cuInit(0);
	}
}
