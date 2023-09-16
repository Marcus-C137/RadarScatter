# RadarScatter
Runs a simulation of a lfm signal generated from a single source using both the cpu and gpu and compares the timing and results

## To Run:
```bash
nvcc test.cu -o test
./test
```

## Visualizer
A visualizer is given too to see the propogating wave. It will show the input signal, the scatter signal, the cpu simulated propagation, and the gpu propagation. It can used via:
```bash
python3 graph.py
```
