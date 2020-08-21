package dezzy.neuronz2.driver.meta;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE;

import java.util.HashMap;
import java.util.Map;

/**
 * Maps CUDA device attributes to short descriptions. The device attributes can be found in {@link jcuda.driver.CUdevice_attribute}.
 * Descriptions are stored in {@link #map}.
 * 
 * @author Joe Desmond
 */
public class AttributeMap {
	
	/**
	 * Maps CUDA device attributes to short descriptions
	 */
	public static final Map<Integer, String> map = getAttributeMap();
	
	/**
	 * Populates the attribute map.
	 * 
	 * @return attribute map
	 */
	private static final HashMap<Integer, String> getAttributeMap() {
		final HashMap<Integer, String> out = new HashMap<Integer, String>();
		
		out.put(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "Maximum number of threads per block");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, "Maximum x-dimension of a block");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, "Maximum y-dimension of a block");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, "Maximum z-dimension of a block");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, "Maximum x-dimension of a grid");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, "Maximum y-dimension of a grid");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, "Maximum z-dimension of a grid");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "Maximum shared memory per thread block in bytes");
		out.put(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, "Total constant memory on the device in bytes");
		out.put(CU_DEVICE_ATTRIBUTE_WARP_SIZE, "Warp size in threads");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_PITCH, "Maximum pitch in bytes allowed for memory copies");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, "Maximum number of 32-bit registers per thread block");
		out.put(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "Clock frequency in kilohertz");
		out.put(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, "Alignment requirement");
		out.put(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "Number of multiprocessors on the device");
		out.put(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, "Whether there is a run time limit on kernels");
		out.put(CU_DEVICE_ATTRIBUTE_INTEGRATED, "Device is integrated with host memory");
		out.put(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, "Device can map host memory into CUDA address space");
		out.put(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, "Compute mode");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, "Maximum 1D texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, "Maximum 2D texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, "Maximum 3D texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, "Maximum 2D texture height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, "Maximum 3D texture height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, "Maximum 3D texture depth");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, "Maximum 2D layered texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, "Maximum 2D layered texture height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, "Maximum layers in a 2D layered texture");
		out.put(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, "Alignment requirement for surfaces");
		out.put(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, "Device can execute multiple kernels concurrently");
		out.put(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, "Device has ECC support enabled");
		out.put(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, "PCI bus ID of the device");
		out.put(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, "PCI device ID of the device");
		out.put(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, "Device is using TCC driver model");
		out.put(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, "Peak memory clock frequency in kilohertz");
		out.put(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, "Global memory bus width in bits");
		out.put(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "Size of L2 cache in bytes");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "Maximum resident threads per multiprocessor");
		out.put(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, "Number of asynchronous engines");
		out.put(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, "Device shares a unified address space with the host");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, "Maximum 1D layered texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, "Maximum layers in a 1D layered texture");
		out.put(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, "PCI domain ID of the device");
		out.put(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, "Pitch alignment requirement for textures");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, "Maximum cubemap texture width/height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, "Maximum cubemap layered texture width/height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, "Maximum layers in a cubemap layered texture");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, "Maximum 1D surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, "Maximum 2D surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, "Maximum 2D surface height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, "Maximum 3D surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, "Maximum 3D surface height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, "Maximum 3D surface depth");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, "Maximum 1D layered surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, "Maximum layers in a 1D layered surface");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, "Maximum 2D layered surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, "Maximum 2D layered surface height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, "Maximum layers in a 2D layered surface");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, "Maximum cubemap surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, "Maximum cubemap layered surface width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, "Maximum layers in a cubemap layered surface");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, "Maximum 1D linear texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, "Maximum 2D linear texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, "Maximum 2D linear texture height");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, "Maximum 2D linear texture pitch in bytes");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, "Maximum mipmapped 2D texture width");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, "Maximum mipmapped 2D texture height");
		out.put(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "Major compute capability version number");
		out.put(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "Minor compute capability version number");
		out.put(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, "Maximum mipmapped 1D texture width");
		out.put(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, "Device supports stream priorities");
		out.put(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, "Device supports caching globals in L1");
		out.put(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, "Device supports caching locals in L1");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "Maximum shared memory per multiprocessor in bytes");
		out.put(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "Maximum number of 32-bit registers per multiprocessor");
		out.put(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, "Device can allocate managed memory on this system");
		out.put(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, "Device is on a multi-GPU board");
		out.put(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, "Unique ID for a group of devices on the same multi-GPU board");
		out.put(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, "Link between the device and the host supports native atomic operations");
		out.put(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, "Ratio of single precision to double precision performance");
		out.put(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, "Device supports pageable memory without calling cudaHostRegister()");
		out.put(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, "Device can coherently access managed memory concurrently with the CPU");
		out.put(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, "Device supports compute preemption");
		
		return out;
	}
}
