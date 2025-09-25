
# Buddy Compiler LLaMA Example

1. Generate computational graphs and parameter files

All MLIR files and parameter files must be generated first. Please refer to [LlaMa/examples/SplitLlama at main · FloatingcloudKnight/LlaMa](https://github.com/FloatingcloudKnight/LlaMa/tree/main/examples/SplitLlama)  

   
  
  
2. Modify the CMakeLists file

Since the distributed system utilizes websocketpp, please ensure that a compiled Boost library and websocketpp header files are present on your computer before compiling. Additionally, modify the following two paths in the CMakeLists file:

```
set(Boost_INCLUDE_DIR "/YOUR_BOOST_PATH")
include_directories(/YOUR_WEBSOCKETPP_PATH)
```



Next, you need to modify the following paths, which are are used to read  parameters after the device starts running.

```
set(LLAMA_SPLIT_EXAMPLE_PATH "/YOUR_RUNTIME_PATH")
set(LLAMA_EXAMPLE_BUILD_PATH "/YOUR_RUNTIME_PATH")
```



The current model employs a fixed distributed communication scheme where a single device handles both input and output (Device 1). Subsequently, every two devices jointly process one or multiple layers of the model. Assuming Device 1 handles input and output, Devices 2 and 3 process the first 16 layers, while Devices 4 and 5 process the last 16 layers.

Reference: Each device requires approximately 3.1 GB of storage space to run the model's 8 layers (more precisely, half of 8 layers). The runtime memory usage is approximately 4 GB.



After determining the number of devices and assigning responsibilities, modify the IP addresses and port numbers of the devices in cmakelisit. IP addresses must use IPv6.

```
set(DEVICEn "DEVICE_IP")
```

Modules requiring segmentation, such as MHA and MLP, generate a separate program for each device. Therefore, the number of compiled programs and their suffixes must be configured. For example, two sets of devices:

```
set(PART_NUMS 2)
set(RUNNER_IDS 0 1)
```



3. Configure the environment and compile the program

   Please refer to  [buddy-benchmark/benchmarks/DeepLearning at main · buddy-compiler/buddy-benchmark](https://github.com/buddy-compiler/buddy-benchmark/tree/main/benchmarks/DeepLearning)

   

   Compile the program using the ninja command, as shown below:：

```
ninja buddy-llama-all-runs # Generate all programs at once
ninja buddy-llama-mha-run  # Compiling a set of programs generated from one of the modules
ninja buddy-llama-mha-run-1 # Compiling a single program
```

