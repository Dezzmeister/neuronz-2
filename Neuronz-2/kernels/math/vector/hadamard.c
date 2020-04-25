__kernel void vectorHadamard(__global const double* vec0, __global const double* vec1, __global double* result) {
	int i = get_global_id(0);
	
	result[i] = vec0[i] * vec1[i];
}