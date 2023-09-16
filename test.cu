#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_PULSES 10
#define PULSE_WIDTH 1000
#define LIGHT_SPEED 3e8 

#define SAMPLE_RATE 1e9 // Sample rate in Hz
#define DURATION 0.0001 // Duration of the LFM signal in seconds
#define SIGNAL_LENGTH DURATION*LIGHT_SPEED
#define DUTY 0.1 // Duty cycle
#define F_START 1e5 // Start frequency in Hz
#define F_END 5e5 // End frequency in Hz

#define GRID_SIZE 250
#define CELL_LENGTH SIGNAL_LENGTH/50 // length of cell in meters
#define TIME_STEPS 200 // Number of time steps in simulation
#define TIME_DELTA 1e-6 // Time difference in seconds between timesteps
#define NUM_SCATTERS 70 // Number of scatter points 
#define LOOK_ANGLE 0.3491 // 30 degrees in radians
#define SWATH_WIDTH 70 // swath width in units of cells 
#define HEIGHT GRID_SIZE //signal starts at top of grid - units in cells

#define BLOCK_SIZE 16

// Define a macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t cudaError = call; \
        if (cudaError != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaError)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


int storeSignal(float* signal, int num_samples, const char* name) {
	
	FILE *signalFile = fopen(name, "w");

	if (signalFile == NULL){
		perror("Error opening file");
		return 1;
	}

    // Output the LFM signal 
    for (int i = 0; i < num_samples; i++) {
        fprintf(signalFile, "%.2lf\n", signal[i]);
    }
	return 0;

}

int get_prop_offset(int x, int y, int t){
    return t * GRID_SIZE * GRID_SIZE + y * GRID_SIZE + x;
}

void write_propagation(float* propagation, const char* filename){
    FILE* file = fopen(filename, "w");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int t = 0; t < TIME_STEPS; t++) {
        fprintf(file, "Time Step %d:\n", t);
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                fprintf(file, "%.2f ", propagation[get_prop_offset(x,y,t)]);
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
}

//makes the linearly frequency modulated signal
void makeLFM(float* signal, int num_samples){

	int non_zero_samples = num_samples * DUTY;
    float delta_f = (F_END - F_START) / (DURATION*DUTY); // Frequency change rate

    //write lfm values
    for (int i = 0; i < non_zero_samples; i++) {
        float t = (float)i / SAMPLE_RATE; // Time in seconds

        // Calculate the instantaneous frequency at time t
        float instant_freq = F_START + delta_f * t;

        // Generate the LFM signal by integrating the instantaneous frequency
        signal[i] = sin(2 * M_PI * instant_freq * t);
    }

    //shift pulse so that it is at the end
    //helps so simulation is not blank at start
    int startInd = num_samples - non_zero_samples;
    for (int i = 0; i < non_zero_samples; i++){
        signal[i+startInd] = signal[i];
        signal[i] = 0;
    }
}

//reverses the linearly frequency modulated signal
//this is the signal that the point scatters emmit  
void reverseLFM(float* signal, int num_samples){
    for (int i = 0; i < num_samples/2; i++){
        float tmp = signal[i];
        signal[i] = signal[num_samples - i - 1];
        signal[num_samples - i - 1] = tmp; 
    }
}

//makes the signal locations based on NUM_SCATTERS, SWATH_WIDTH, and LOOK_ANGLE
void makeScatterCoords(int* coords){
    //find start location of swath
    int offset = HEIGHT * tan(LOOK_ANGLE);
    //find scatter point interval
    float interval = float(SWATH_WIDTH) / float(NUM_SCATTERS+1);
    for (int i = 1; i < NUM_SCATTERS+1; i++){
        coords[i-1] = offset + interval * i;
    }
}

// pass contants in as parameter to avoid slow global constant memory on the gpu
struct consts{
    int cell_length;
    float ligth_speed;
    float time_delta;
    float duration;
    int grid_size;
    int time_steps;
    int num_samples;
    int num_scatters;
} constants;


// Kernel function to add shifted pulses
__global__ void make_propagation_gpu(float* pulses, float* output, int* signal_locations, consts constants) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //This thread will be responsible for the cell at x,y
    if (idx < constants.grid_size && idy < constants.grid_size) {
        int index_xy = idy * constants.grid_size + idx;
        //The thread will fill the cell's values through time
        for (int time_step = 0; time_step < constants.time_steps; time_step++){
            //Add all signals together
            for (int signal_idx = 0; signal_idx < constants.num_scatters; signal_idx++){
                int index = time_step*constants.grid_size*constants.grid_size + index_xy;
                //get distance from point source
                int distance = sqrt(pow(idx-signal_locations[signal_idx],2) + pow(idy,2));
                //find time delay from cell to point source
                float td = (distance * constants.cell_length)/ constants.ligth_speed;
                //get difference between time delay and current time
                float t = time_step*constants.time_delta - td;
                //if cell has not recieved the signal continue
                if (t < 0.0){
                    continue;
                }
                //if time is larger than the signal duration wrap around since the signal is repeated
                t = fmod(t, constants.duration);
                //find where the signal is at at that time
                int element = (t/constants.duration) * float(constants.num_samples);

                //add to the cell at time t the signal divided by the number of scatters
                //i.e amplitude of incomming signal is split among each scatter 
                output[index] += pulses[element]/ constants.num_scatters;
            }
        }
    }
}

int main() {
	int num_samples = (int)(DURATION * SAMPLE_RATE);
    size_t propogation_size = GRID_SIZE * GRID_SIZE *TIME_STEPS * sizeof(float);

	//allocate cpu memory and gpu memory for input and output signals
    float   *h_signal            = (float*)calloc(num_samples, sizeof(float));
    int     *h_signal_locations  = (int*)calloc(NUM_SCATTERS, sizeof(int));
    float   *h_propagation       = (float*)malloc(propogation_size);
    float   *h_output            = (float*)malloc(propogation_size);

    float   *d_signal, *d_output;
    int     *d_signal_locations;
    CUDA_CHECK(cudaMalloc((void**)&d_signal, num_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, propogation_size));
    CUDA_CHECK(cudaMalloc((void**)&d_signal_locations, NUM_SCATTERS * sizeof(int)));

	//Make the radar signal 
	makeLFM(h_signal, num_samples);
	storeSignal(h_signal, num_samples, "signal.txt");
    //Reverse the signal to recreate the scatter signal
    reverseLFM(h_signal, num_samples);
	storeSignal(h_signal, num_samples, "signal_reversed.txt");

    //Make the scatter coordinates
    makeScatterCoords(h_signal_locations);

    clock_t start_cpu_time = clock();

    // for each time step
    for (int time_step = 0; time_step < TIME_STEPS; time_step++){
        // for each cell
        for (int x = 0; x < GRID_SIZE; x++){
            for (int y = 0; y < GRID_SIZE; y++){
                //for each signal
                for (int signal_idx = 0; signal_idx < NUM_SCATTERS; signal_idx++){
                    // get position in propagation array that corresponds to the x,y,t point
                    int offset = get_prop_offset(x, y, time_step);
                    //get distance from point source
                    int distance = sqrt(pow(x-h_signal_locations[signal_idx],2) + pow(y,2));
                    //find time delay from cell to point source
                    float td = (distance * CELL_LENGTH)/ LIGHT_SPEED;
                    //get difference between time delay and current time
                    float t = time_step*TIME_DELTA - td;
                    //if cell has not recieved the signal continue
                    if (t < 0.0){
                        continue;
                    }
                    //if time is larger than the signal duration wrap around since the signal is repeated
                    t = fmod(t, DURATION);
                    //find where the signal is at at that time
                    int element = (t/DURATION) * float(num_samples);
                    if (element >= num_samples){
                        printf("element: %d, num_samples: %d\n", element, num_samples);
                    }
                    //add to the cell at time t the signal divided by the number of scatters
                    //i.e amplitude of incomming signal is split among each scatter 
                    h_propagation[offset] += h_signal[element]/NUM_SCATTERS;
                }
            }
        }
    }

    clock_t stop_cpu_time = clock();
    printf("cpu time: %lf\n", (double)(stop_cpu_time - start_cpu_time) / CLOCKS_PER_SEC);
    
    write_propagation(h_propagation, "cpu_propagation.txt");

    clock_t start_gpu_time = clock();

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal, num_samples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_signal_locations, h_signal_locations, NUM_SCATTERS * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel
    constants.cell_length = CELL_LENGTH;
    constants.ligth_speed = LIGHT_SPEED;
    constants.time_delta = TIME_DELTA;
    constants.duration = DURATION;
    constants.grid_size = GRID_SIZE;
    constants.time_steps = TIME_STEPS;
    constants.num_samples = num_samples;
    constants.num_scatters = NUM_SCATTERS;

    //Make a block of 256 threads with x and y parameters
    //Make a grid with the correct number of blocks so 
    //that each cell in the simulation can have its own thread
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((GRID_SIZE + BLOCK_SIZE - 1)/ BLOCK_SIZE, (GRID_SIZE+BLOCK_SIZE -1) / BLOCK_SIZE);
    
    make_propagation_gpu<<<gridDim, blockDim>>>(d_signal, d_output, d_signal_locations, constants);
    CUDA_CHECK(cudaGetLastError());

    // Copy the result back to CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, propogation_size, cudaMemcpyDeviceToHost));

    clock_t stop_gpu_time = clock();
    printf("gpu time: %lf\n", (double)(stop_gpu_time - start_gpu_time) / CLOCKS_PER_SEC);


	// Store recieved signal
	write_propagation(h_output, "gpu_propagation.txt");

    // Free GPU memory
    cudaFree(d_signal);
    cudaFree(d_output);
    cudaFree(d_signal_locations);

    // Process the result (h_output)

    // Cleanup
    free(h_signal);
    free(h_output);
    free(h_propagation);
    free(h_signal_locations);

    return 0;
}
