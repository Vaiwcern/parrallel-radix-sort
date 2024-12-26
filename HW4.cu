#include <stdio.h>
#include <stdint.h>

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

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

// Kernel 1: Extract bit at position `bitIdx`
__global__ void extract_bits_kernel(uint32_t *a, int *bit, int n, int bitIdx) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        bit[tid] = (a[tid] >> bitIdx) & 1;
    }
}

// Kernel 3: Sort elements based on the current bit
__global__ void sort_by_bit_kernel(uint32_t *a, uint32_t *out, int *bit, int *nOneBefore, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        int numZeros = n - nOneBefore[n - 1] - bit[n - 1];
        int rank;
        if (bit[tid] == 0) {
            rank = tid - nOneBefore[tid];
        } else {
            rank = numZeros + nOneBefore[tid];
        }
        out[rank] = a[tid];
    }
}


__global__ void scanBlkKernel2(int * in, int n, int * out, int * blkSums)
{
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

__global__ void addPrevBlkSum(int * blkSumsScan, int * blkScans, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (i < n)
        blkScans[i] += blkSumsScan[blockIdx.x];
}

void scan(int * d_in1, int n, int * d_out2, dim3 blkSize=dim3(1)) {
    int blkDataSize;
    blkDataSize = 2 * blkSize.x;
    // 1. Scan locally within each block, 
    //    and collect blocks' sums into array
    int * d_blkSums;
    dim3 gridSize((n - 1) / blkDataSize + 1);
    if (gridSize.x > 1)
    {
        CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    }
    else
    {
        d_blkSums = NULL;
    }

    size_t smem = blkDataSize * sizeof(int);
    scanBlkKernel2<<<gridSize, blkSize, smem>>>(d_in1, n, d_out2, d_blkSums);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    if (gridSize.x > 1)
    {
        // 2. Compute each block's previous sum 
        //    by scanning array of blocks' sums
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

    CHECK(cudaFree(d_blkSums));
}

void sortByDevice(const uint32_t *in, int n, uint32_t *out, int blockSize) {
    uint32_t *d_in, *d_out;
    int *d_bit, *d_nOneBefore;

    // Allocate device memory
    cudaMalloc(&d_in, n * sizeof(uint32_t));
    cudaMalloc(&d_out, n * sizeof(uint32_t));
    cudaMalloc(&d_bit, n * sizeof(int));
    cudaMalloc(&d_nOneBefore, (n + 1) * sizeof(int));

    cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int bitIdx = 0; bitIdx < 32; ++bitIdx) {
        // Step 1: Extract bits (kernel)
        extract_bits_kernel<<<numBlocks, blockSize>>>(d_in, d_bit, n, bitIdx);
        cudaDeviceSynchronize();

        // Step 2: Perform exclusive scan sequentially on host
        cudaMemset(d_nOneBefore, 0, sizeof(int));

        dim3 blkSize = dim3(blockSize);
        scan(d_bit, n, d_nOneBefore + 1, blkSize);

        // Step 3: Sort elements based on the current bit (kernel)
        sort_by_bit_kernel<<<numBlocks, blockSize>>>(d_in, d_out, d_bit, d_nOneBefore, n);
        cudaDeviceSynchronize();

        // Swap input and output arrays for the next iteration
        uint32_t *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // Copy sorted result back to host
    cudaMemcpy(out, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_bit);
    cudaFree(d_nOneBefore);
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
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

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
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

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    // int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        in[i] = rand() % 255; // For test by eye
        // in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
