import platform
import psutil
import tensorflow as tf

# Get system information
system_info = platform.uname()

# Print basic system information
print("System Information:")
print(f"System: {system_info.system}")
print(f"Node Name: {system_info.node}")
print(f"Release: {system_info.release}")
print(f"Version: {system_info.version}")
print(f"Machine: {system_info.machine}")
print(f"Processor: {system_info.processor}")

# Get CPU information
cpu_info = platform.processor()
cpu_count = psutil.cpu_count(logical=False)  # Physical CPU cores
cpu_threads = psutil.cpu_count(logical=True)  # Total CPU threads

# Print CPU information
print("\nCPU Information:")
print(f"Processor: {cpu_info}")
print(f"Physical Cores: {cpu_count}")
print(f"Total Threads: {cpu_threads}")

# Get RAM (Memory) information
ram_info = psutil.virtual_memory()

# Print RAM information
print("\nRAM (Memory) Information:")
print(f"Total RAM: {ram_info.total / (1024 ** 3):.2f} GB")
print(f"Available RAM: {ram_info.available / (1024 ** 3):.2f} GB")
print(f"Used RAM: {ram_info.used / (1024 ** 3):.2f} GB")
print(f"Memory Usage Percentage: {ram_info.percent:.2f}%")

# Get GPU information (if available)
gpu_info = tf.config.experimental.list_physical_devices('GPU')

# Print GPU information (if GPU is available)
if gpu_info:
    print("\nGPU Information:")
    for idx, gpu in enumerate(gpu_info):
        print(f"GPU {idx + 1}: {gpu.name}")
else:
    print("\nNo GPU available in this environment.")

# TensorFlow version
tf_version = tf.__version__

# Print TensorFlow version
print("\nTensorFlow Version:", tf_version)
