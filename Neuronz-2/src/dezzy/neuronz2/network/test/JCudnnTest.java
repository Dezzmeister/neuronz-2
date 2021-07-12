package dezzy.neuronz2.network.test;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.*;
import static jcuda.jcudnn.cudnnConvolutionMode.*;
import static jcuda.jcudnn.cudnnDataType.*;
import static jcuda.jcudnn.cudnnStatus.*;
import static jcuda.jcudnn.cudnnTensorFormat.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import java.io.IOException;
import java.nio.ByteBuffer;

import dezzy.neuronz2.cuda.math.GVector;
import dezzy.neuronz2.dataio.DoubleImage;
import dezzy.neuronz2.driver.Init;
import jcuda.Pointer;
import jcuda.driver.CUresult;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdAlgo;
import jcuda.jcudnn.cudnnConvolutionFwdAlgoPerf;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnStatus;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

public class JCudnnTest {

	public static void main(final String[] args) throws IOException {
		final DoubleImage imageIn = new DoubleImage("data/images/tee.png");
		final GVector imageVec = new GVector(imageIn.toVector());
		
		int width = imageIn.width;
		int height = imageIn.height;
		
		JCuda.setExceptionsEnabled(true);	
		
		int statusCode = Init.CUDA_DRIVER_STATUS_CODE;
		if (statusCode != CUresult.CUDA_SUCCESS) {
			System.out.println("Something went wrong while initializing CUDA");
			System.exit(-1);
		}
		
		System.out.println("CUDA initialized");
		
		cudnnHandle handle = new cudnnHandle();		
		checkCudnn(cudnnCreate(handle));
		
		cudnnTensorDescriptor inputDescriptor = new cudnnTensorDescriptor();
		checkCudnn(cudnnCreateTensorDescriptor(inputDescriptor));
		checkCudnn(cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, 1, 1, height, width));
		
		cudnnTensorDescriptor outputDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(outputDescriptor);
		cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, 1, 1, height, width);
		
		cudnnFilterDescriptor kernelDescriptor = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(kernelDescriptor);
		cudnnSetFilter4dDescriptor(kernelDescriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, 1, 3, 3);
		
		cudnnConvolutionDescriptor convolutionDescriptor = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convolutionDescriptor);
		cudnnSetConvolution2dDescriptor(convolutionDescriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);
		
		cudnnConvolutionFwdAlgoPerf convAlgorithm;
		
		int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
		int returnedAlgoCount = -1;
		int[] returnedAlgoCountArray = {0};
		cudnnConvolutionFwdAlgoPerf[] results = new cudnnConvolutionFwdAlgoPerf[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
		System.out.println("cudnnFindConvolutionForwardAlgorithm");
		//cudnnGetConvolutionForwardAlgorithm_v7(handle, inputDescriptor, kernelDescriptor, convolutionDescriptor, outputDescriptor, requestedAlgoCount, returnedAlgoCountArray, results);
		cudnnFindConvolutionForwardAlgorithm(handle, inputDescriptor, kernelDescriptor, convolutionDescriptor, outputDescriptor, requestedAlgoCount, returnedAlgoCountArray, results);
		returnedAlgoCount = returnedAlgoCountArray[0];
		
		for (int i = 0; i < returnedAlgoCount; i++) {
			String errorString = cudnnGetErrorString(results[i].status);
			System.out.printf("^^^^ %s for Algo %s: %f time requiring %d memory\n", errorString, cudnnConvolutionFwdAlgo.stringFor(results[i].algo), results[i].time, (long)results[i].memory);
		}
		
		convAlgorithm = results[0];
		
		long[] workspaceBytes = new long[1];
		cudnnGetConvolutionForwardWorkspaceSize(handle, inputDescriptor, kernelDescriptor, convolutionDescriptor, outputDescriptor, convAlgorithm.algo, workspaceBytes);
		
		System.out.println("Workspace size: " + (workspaceBytes[0]) + " bytes"); // Should match convAlgorithm.memory
		
		Pointer workspacePtr = new Pointer();
		cudaMalloc(workspacePtr, workspaceBytes[0]);
		
		int[] nPtr = new int[1];
		int[] cPtr = new int[1];
		int[] hPtr = new int[1];
		int[] wPtr = new int[1];
		
		cudnnGetConvolution2dForwardOutputDim(convolutionDescriptor, inputDescriptor, kernelDescriptor, nPtr, cPtr, hPtr, wPtr);
		
		int n = nPtr[0];
		int c = cPtr[0];
		int h = hPtr[0];
		int w = wPtr[0];
		
		int imageBytes = n * c * h * w * 8; // batchSize * channels * height * width * sizeof(double)
		System.out.println("(n, c, h, w): " + "(" + n + ", " + c + ", " + h + ", " + w + ")");
		System.out.println("imageBytes: " + imageBytes);
		
		Pointer srcData = new Pointer();
		Pointer dstData = new Pointer();
		
		Pointer imgDataPtr = imageVec.getPointer();
		
		cudaMalloc(srcData, imageBytes);
		cudaMemcpy(srcData, imgDataPtr, imageBytes, cudaMemcpyHostToDevice);
		
		cudaMalloc(dstData, imageBytes);
		
		double[] kernel = {
			1.0, 1.0, 1.0,
			1.0, -8.0, 1.0,
			1.0, 1.0, 1.0
		};
		
		Pointer hostKernelPtr = Pointer.to(kernel);
		Pointer kernelData = new Pointer();
		int kernelBytes = 9 * 8;
		
		cudaMalloc(kernelData, kernelBytes);
		cudaMemcpy(kernelData, hostKernelPtr, kernelBytes, cudaMemcpyHostToDevice);
		
		Pointer alpha = Pointer.to(new double[] {1.0});
		Pointer beta = Pointer.to(new double[] {0.0});
		
		cudnnConvolutionForward(handle, alpha, inputDescriptor, srcData, kernelDescriptor, kernelData, convolutionDescriptor, convAlgorithm.algo, workspacePtr, workspaceBytes[0], beta, outputDescriptor, dstData);		
		
		double[] convOut = new double[imageBytes / 8];
		Pointer convOutPtr = Pointer.to(convOut);
		
		cudaMemcpy(convOutPtr, dstData, imageBytes, cudaMemcpyDeviceToHost);
		
		DoubleImage outImg = new DoubleImage(convOut, width, height);
		outImg.save("data/images/convtest1.png", "PNG");
		
		cudaFree(kernelData);
		cudaFree(srcData);
		cudaFree(dstData);
		cudaFree(workspacePtr);
		cudnnDestroyTensorDescriptor(inputDescriptor);
		cudnnDestroyTensorDescriptor(outputDescriptor);
		cudnnDestroyFilterDescriptor(kernelDescriptor);
		cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
		cudnnDestroy(handle);
	}
	
	static void checkCudnn(int statusCode) {
		if (statusCode != CUDNN_STATUS_SUCCESS) {
			throw new RuntimeException(cudnnStatus.stringFor(statusCode));
		}
	}

}
