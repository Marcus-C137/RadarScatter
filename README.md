# RadarScatter
Runs a simulation of a lfm signal generated from a single source using both the cpu and gpu and compares the timing and results for each

## To Run:
```bash
nvcc test.cu -o test.o
./test.o
```

## Visualizer
A visualizer is given too to see the propagating wave. It will show the input signal, the scatter signal, the cpu simulated propagation, and the gpu propagation. It can used via:
```bash
python3 graph.py
```
## CurrentBenchMarks:
GridSize: 250 x 250 TimeSteps: 200 ScatterPoints: 70  (875,000,000 operations) <br>
Cpu: 28.43 seconds <br>
Gpu: 0.988 <br>
Hardware: GEforce 3060 <br>
