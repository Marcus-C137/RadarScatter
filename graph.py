import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap

def plotSignal():
    # Read data from the first file into a NumPy array
    data = np.loadtxt("signal.txt")

    # Create an array of indices for the first plot
    indices = np.arange(len(data))

    # Create a subplot with two plots stacked vertically
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.plot(indices, data)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Input LFM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Ensures plots do not overlap
    plt.show()

    # Read data from the first file into a NumPy array
    data = np.loadtxt("signal_reversed.txt")

    # Create an array of indices for the first plot
    indices = np.arange(len(data))

    # Create a subplot with two plots stacked vertically
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.plot(indices, data)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Scatter LFM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Ensures plots do not overlap
    plt.show()

def plotPropogation(filename):

    #read in propogation data
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    current_time_step = None

    for line in lines:
        if line.startswith("Time Step"):
            current_time_step = []
            data.append(current_time_step)
        elif current_time_step is not None:
            values = [float(val) for val in line.strip().split()]
            current_time_step.append(values)
    
    data = np.array(data)

    def update(val):
        im.set_data(data[slider.val])
        print(f"updating data from indx {slider.val}")
        fig.canvas.draw_idle()

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    colors = [(1, 1, 0), (0, 0, 0), (0, 1, 0)]  # Yellow, Black, Green
    cmap_name = 'custom_colormap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    im = ax.imshow(data[0], cmap=custom_cmap, origin="lower", vmin=-.5, vmax=.5, 
                   extent=[0, data.shape[2], 0, data.shape[1]])
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Parameter', 0, len(data) - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    plt.title(filename)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def main():
    plotSignal()
    plotPropogation("cpu_propagation.txt")
    plotPropogation("gpu_propagation.txt")

if __name__ == "__main__":
    main()
