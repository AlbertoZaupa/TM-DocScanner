import matplotlib.pyplot as plt
import sys

filename = "benchmarks"
median_blur_times = []
edge_detection_times = []
page_frame_detection_times = []
binarization_times = []

with open(filename, "r") as f:
    i = 0
    for line in f:
        time = float(line[:-1])
        if i % 4 == 0:
            median_blur_times.append(time)
        elif i % 4 == 1:
            edge_detection_times.append(time)
        elif i % 4 == 2:
            page_frame_detection_times.append(time)
        else:
            binarization_times.append(time)
        i += 1

x_axis = range(int(i / 4))
plt.plot(x_axis, median_blur_times, color="blue", label="median blur")
plt.plot(x_axis, edge_detection_times, color="red", label="edge detect")
plt.plot(x_axis, page_frame_detection_times, color="orange", label="page frame")
plt.plot(x_axis, binarization_times, color="green", label="bin")
plt.legend(loc='upper right')
plt.savefig("./times_plot.jpg")
plt.show()
