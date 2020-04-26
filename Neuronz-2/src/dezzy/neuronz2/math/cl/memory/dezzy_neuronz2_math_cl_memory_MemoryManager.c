#include <stdlib.h>
#include "dezzy_neuronz2_math_cl_memory_MemoryManager.h"

JNIEXPORT jdoubleArray JNICALL Java_dezzy_neuronz2_math_cl_memory_MemoryManager_alignedMalloc (JNIEnv * env, jobject obj, jlong size, jlong alignment) {
	void* ptr = _aligned_malloc(size * sizeof(double), alignment);
	return (jdoubleArray) ptr;
}

JNIEXPORT void JNICALL Java_dezzy_neuronz2_math_cl_memory_MemoryManager_alignedFree (JNIEnv * env, jobject obj, jdoubleArray ptr) {
	_aligned_free((void*) ptr);
}
	