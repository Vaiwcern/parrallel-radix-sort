#include <stdio.h>
#include <iostream> 

using namespace std;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start); 
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

/*
Scan within each block's data (work-efficient), write results to "out", and
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
__global__ void scanBlkKernel2(int * in, int n, int * out, int * blkSums)
{
    // TODO
	// 1. Each block loads data from GMEM to SMEM
	extern __shared__ int s_data[];
	int i1 = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	int i2 = i1 + blockDim.x;
	if (i1 < n)
		s_data[threadIdx.x] = in[i1];
	if (i2 < n)
		s_data[threadIdx.x + blockDim.x] = in[i2];
	__syncthreads();

	// 2. Each block does scan with data on SMEM
	// 2.1. Reduction phase
	for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}
	// 2.2. Post-reduction phase
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride; // Wow
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}

	// 3. Each block writes results from SMEM to GMEM
	if (i1 < n)
		out[i1] = s_data[threadIdx.x];
	if (i2 < n)
		out[i2] = s_data[threadIdx.x + blockDim.x];

	if (blkSums != NULL && threadIdx.x == 0)
		blkSums[blockIdx.x] = s_data[2 * blockDim.x - 1];
}

// TODO: You can define necessary functions here
__global__ void addPrevBlkSum(int * blkSumsScan, int * blkScans, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (i < n)
        blkScans[i] += blkSumsScan[blockIdx.x];
}

void scan(int * in1, int n, int * out1, dim3 blkSize=dim3(1)) {
    int blkDataSize;
    printf("\nScan by device, work-efficient\n");
    blkDataSize = 2 * blkSize.x;
    // 1. Scan locally within each block, 
    //    and collect blocks' sums into array
    
    int * d_in1, * d_out2, * d_blkSums;
    size_t nBytes = n * sizeof(int);
    CHECK(cudaMalloc(&d_in1, nBytes)); 
    CHECK(cudaMalloc(&d_out2, nBytes)); 
    dim3 gridSize((n - 1) / blkDataSize + 1);
    if (gridSize.x > 1)
    {
        CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    }
    else
    {
        d_blkSums = NULL;
    }

    CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));

    size_t smem = blkDataSize * sizeof(int);
    scanBlkKernel2<<<gridSize, blkSize, smem>>>(d_in1, n, d_out2, d_blkSums);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    if (gridSize.x > 1)
    {
        // 2. Compute each block's previous sum 
        //    by scanning array of blocks' sums
        // TODO
        size_t temp = gridSize.x * sizeof(int);
        int * blkSums = (int*)malloc(temp);
        CHECK(cudaMemcpy(blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridSize.x; i++)
            blkSums[i] += blkSums[i-1];
        CHECK(cudaMemcpy(d_blkSums, blkSums, temp, cudaMemcpyHostToDevice));

        // 3. Add each block's previous sum to its scan result in step 1
        addPrevBlkSum<<<gridSize.x - 1, blkDataSize>>>(d_blkSums, d_out2, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        
        free(blkSums);
    }

    CHECK(cudaMemcpy(out1, d_out2, nBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_out2));
    CHECK(cudaFree(d_blkSums));
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(int * out, int * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = 5;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(int);
    int * in = (int *)malloc(bytes);
    int * out = (int *)malloc(bytes); // Device result
    int * correctOut = (int *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = (int)(rand() & 0xFF); // random int in [-127, 128]

    // DETERMINE BLOCK SIZE
    dim3 blockSize1(512); 
    dim3 blockSize2(256); 
    if (argc == 3)
    {
        blockSize1.x = atoi(argv[1]);
        blockSize2.x = atoi(argv[2]);
    }

    for (int i = 0; i < n; ++i) {
        cout << in[i] << "\n";
    }

    // SCAN BY DEVICE, WORK-EFFICIENT
    memset(out, 0, n * sizeof(int)); // Reset out
    scan(in, n, out, blockSize2);

    for (int i = 0; i < n; ++i) {
        cout << out[i] << "\n";
    }

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
