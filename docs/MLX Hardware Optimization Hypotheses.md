# **Deep Research Directive: Hardware-Aware MLX Optimizations for Autonomous Research Loops**

The optimization of neural network architectures for specialized hardware necessitates a paradigm shift from theoretical algorithmic complexity to hardware-sympathetic execution. In the context of the corridorkey-mlx neural green screen unmixing engine, ported to Apple Silicon via the MLX framework, standard PyTorch-derived inference strategies frequently fail to leverage the unique microarchitectural features of the M-series processors. This report establishes a comprehensive theoretical foundation and experimental framework for an autonomous, Large Language Model (LLM)-driven autoresearch loop. The objective is to continuously mutate the inference code to minimize Peak Active Memory and Median Execution Time.

The ensuing analysis is rooted strictly in High-Performance Computing (HPC) principles, specifically those detailing Memory-Level Parallelism (MLP), cache hierarchy utilization, branchless programming, and Just-In-Time (JIT) graph compilation. By aligning the CNNRefiner and associated subsystems with the underlying physics of Apple's Unified Memory Architecture (UMA) and the Metal Performance Shaders (MPS) backend, the autonomous agent can systematically test and validate highly advanced optimizations.

## **Cache Locality and Hardware-Aligned Tiling (UMA)**

The Apple Silicon architecture represents a radical departure from traditional discrete central processing unit (CPU) and graphics processing unit (GPU) configurations. The Unified Memory Architecture (UMA) allows the CPU, GPU, Neural Engine (ANE), and Advanced Matrix Extensions (AMX) coprocessor to access a single pool of high-bandwidth memory.1 While this eliminates the latency of copying data across a peripheral component interconnect express (PCIe) bus, retrieving data from the main unified memory pool remains orders of magnitude slower than accessing on-chip Static Random-Access Memory (SRAM).3

To achieve optimal throughput, intermediate activations generated during the inference of a convolutional neural network must be retained within the L2 cache or the System Level Cache (SLC). When a tensor's working set size exceeds these cache boundaries, the processor is forced into a state of cache thrashing, leading to costly evictions to main memory.3 The autonomous research loop must therefore treat spatial tiling not merely as a mechanism to avoid Out-Of-Memory (OOM) errors, but as a primary vector for hardware alignment.

### **Apple Silicon Cache Hierarchies**

Understanding the exact memory boundaries of the target hardware is critical for formulating tiling dimensions. The M-series chips possess distinct cache topologies depending on their generation and tier. The performance engineering principles outlined in Sergey Slotin's methodologies dictate that algorithms must be shaped to fit the physical geometry of the hardware.3

| System-on-Chip (SoC) | Performance Core L2 Cache | Efficiency Core L2 Cache | System Level Cache (SLC) | Memory Bandwidth |
| :---- | :---- | :---- | :---- | :---- |
| **Apple M1 / M2** | 12 MB – 16 MB | 4 MB | 8 MB – 24 MB | 67 GB/s – 100 GB/s |
| **Apple M3 Max** | 32 MB | 4 MB | 48 MB | Up to 400 GB/s |
| **Apple M4 Pro** | 16 MB – 32 MB | 4 MB | Up to 48 MB | 273 GB/s |
| **Apple M4 Max** | 32 MB | 4 MB | Up to 96 MB | 410 GB/s – 546 GB/s |

Data derived from architectural analyses of Apple Silicon processor specifications.5

Currently, the CNNRefiner processes the image in arbitrary ![][image1] tiles. However, a ![][image1] tile with 64 channels in 32-bit floating-point (FP32) precision requires a significant memory footprint. The memory required for a single such tensor is calculated as:

![][image2]  
This ![][image3] footprint vastly exceeds the ![][image4] L2 cache of even the most advanced M4 Max processor.6 During a single forward pass of the CNNRefiner, which consists of multiple sequential convolutions, normalizations, and activations, the total active memory required to hold the input, the output, and the intermediate feature maps expands well beyond this figure. Consequently, the intermediate activation buffers are repeatedly flushed to the main UMA pool, stalling the arithmetic logic units (ALUs) while waiting for memory fetches.

### **Hypothesis 1: L2-Aligned Spatial Tiling**

The first major hypothesis to be formulated for the autonomous agent is the dynamic calculation of hardware-sympathetic tile sizes. The autoresearch loop must dynamically adjust the tile\_size parameter so that the maximum memory footprint of the deepest layer in the CNNRefiner remains strictly below the L2 cache capacity of the host machine's Performance (P) cores.

By enforcing a constraint where the Working Set Size (WSS) is maintained at ![][image5] (reserving twenty percent for instruction caches and operating system overhead), the intermediate tensors will never leave the SRAM. The MLX compiler can then fuse operations seamlessly without allocating intermediate UMA buffers.8

The autonomous agent should be instructed to inject the following code mutation into the tiling.py module. This implementation dynamically derives the optimal tile size based on the target machine's microarchitecture.

Python

import mlx.core as mx  
import math

def calculate\_optimal\_tile\_size(channels: int, dtype\_bytes: int \= 4, target\_cache\_mb: float \= 16.0) \-\> int:  
    """  
    Hypothesis 1: Calculate the maximum spatial dimension (H \= W) that fits   
    entirely within the L2 cache bound to prevent UMA roundtrips.  
    The agent will mutate \`target\_cache\_mb\` during the search phase.  
    """  
    target\_cache\_bytes \= target\_cache\_mb \* 1024 \* 1024  
      
    \# Account for input, output, and at least 2 intermediate activation buffers  
    \# in the deepest segment of the CNNRefiner graph.  
    num\_concurrent\_buffers \= 4   
      
    \# Equation: H \* W \* C \* bytes \* buffers \<= Cache Size  
    \# Since H \= W for square tiles: W^2 \= Cache Size / (C \* bytes \* buffers)  
    max\_pixels \= target\_cache\_bytes / (channels \* dtype\_bytes \* num\_concurrent\_buffers)  
    optimal\_dim \= int(math.sqrt(max\_pixels))  
      
    \# Snap to the nearest multiple of 16 for Metal Threadgroup alignment  
    \# Unaligned memory accesses cause hardware bank conflicts.  
    aligned\_dim \= (optimal\_dim // 16) \* 16  
    return aligned\_dim

class HardwareSympatheticTiler:  
    def \_\_init\_\_(self, hardware\_l2\_mb: float \= 16.0):  
        self.tile\_size \= calculate\_optimal\_tile\_size(channels=64, target\_cache\_mb=hardware\_l2\_mb)  
        \# 25% overlap to maintain receptive field continuity at boundaries  
        self.overlap \= self.tile\_size // 4 

    def process\_image(self, image: mx.array, refiner\_model) \-\> mx.array:  
        """  
        The autoresearch loop evaluates the runtime of this method.   
        A drop in Median Execution Time indicates successful L2 cache residency.  
        """  
        \# Tiling logic implementation...  
        pass

The autoresearch loop must run a parameter sweep on target\_cache\_mb, testing boundaries from ![][image6] (Efficiency Core limits) up to ![][image7] (M4 Max SLC limits). The hypothesis anticipates a non-linear performance curve where the Median Execution Time drops precipitously exactly when the tile size crosses below the physical cache threshold of the execution hardware.

### **Hypothesis 2: SLC-Aware Overlap Striding and Z-Order Curves**

When spatial tiling involves overlapping boundaries to prevent edge artifacts in the CNNRefiner, redundant computations are performed. Standard implementations iterate over tiles in a linear, row-major scanline order. However, scanline traversal exhibits poor 2D spatial locality. By the time the algorithm reaches the tile directly below the current one, the overlapping boundary data from the upper tile has long been evicted from the cache.3

The agent should hypothesize an execution schedule where adjacent tiles are processed utilizing a space-filling curve, such as a Z-order curve (Morton code) or a Hilbert curve. Because the System Level Cache (SLC) can hold up to ![][image8] on Ultra and Max tiers 5, processing overlapping tiles in a tightly localized 2D cluster ensures that the overlapping boundary data remains resident in the SLC.

The experimental mutation requires the agent to modify the dispatch order of the tiling generator. Instead of a standard nested for loop, the agent will implement a Morton code interleaving algorithm to generate the tile coordinates. This maximizes cache line hits, ensuring that as the Metal GPU threads transition to the next tile, the overlapped pixel activations are retrieved instantly from the SLC rather than incurring the latency of a main memory read. The autoresearch agent must benchmark this space-filling curve traversal against the baseline scanline traversal, monitoring the Peak Active Memory profile to ensure the MLX graph evaluator handles the non-linear dispatch without allocating massive intermediate buffer states.

## **Branchless Programming for sRGB-to-Linear Color Math**

The mathematical translation of image data from the display-referred sRGB color space to a linear light space is a mandatory precursor for mathematically accurate blending, convolutions, and neural unmixing. corridorkey-mlx operates in a physically based rendering context, requiring absolute linearity before neural processing.

The sRGB transfer function is a piecewise mathematical operation featuring a linear toe for extreme shadows and an exponential curve for the midtones and highlights.10 The precise International Electrotechnical Commission (IEC) 61966-2-1 specification for sRGB to Linear conversion is defined as:

![][image9]  
In conventional deep learning frameworks, this piecewise discontinuity is typically handled using boolean masks, resulting in code resembling mx.where(tensor \<= 0.04045, linear\_path, gamma\_path). However, from a high-performance computing perspective, boolean masks and conditional branches introduce profound hardware inefficiencies.4

Modern GPUs and the Apple AMX coprocessor operate on Single Instruction, Multiple Data (SIMD) or Single Instruction, Multiple Threads (SIMT) architectures.3 When a where clause is executed, it forces thread divergence: the compute unit must evaluate both branches of the conditional logic across the vector, and the framework must selectively mask the output. Furthermore, materializing a boolean tensor of the same spatial dimensions as a ![][image10] video frame consumes unnecessary memory bandwidth and disrupts the MLX compiler's ability to fuse adjacent kernels. The computational cost of branching is severe; a pipeline flush caused by branch divergence takes orders of magnitude longer to resolve than a simple continuous mathematical operation.3

To optimize corridorkey-mlx, the autoresearch agent must explore branchless programming techniques derived from low-level systems engineering.

### **Hypothesis 3: Pure Algebraic Blending**

The agent must rewrite the piecewise sRGB-to-Linear function in MLX using pure mathematical blending. By converting the logical condition into an arithmetic multiplier consisting solely of 1.0 and 0.0, the entire operation can be fused into a single, contiguous Metal instruction without branching or boolean mask materialization.

In MLX, casting a logical operation directly to a float (mx.astype) transforms the boolean evaluation into an integer or float representation.13 Because MLX compiles the graph via mx.compile 8, this algebraic formulation allows the compiler to generate a branchless predicated sequence of operations on the GPU, avoiding the overhead of mx.where.

The autoresearch loop should inject the following branchless implementation into the preprocessing pipeline and evaluate the execution time reduction.

Python

import mlx.core as mx

@mx.compile  
def branchless\_srgb\_to\_linear(srgb: mx.array) \-\> mx.array:  
    """  
    Hypothesis 3: Execute the sRGB conversion without pipeline divergence.  
    Avoids mx.where() by mathematically blending the two piecewise states.  
    """  
    \# Define IEC 61966-2-1 constants  
    threshold \= 0.04045  
    linear\_divisor \= 12.92  
    gamma\_offset \= 0.055  
    gamma\_divisor \= 1.055  
    gamma\_power \= 2.4

    \# Calculate both paths concurrently. This is highly SIMD friendly as there  
    \# is no instruction pointer divergence across the GPU threadgroup.  
    linear\_path \= srgb / linear\_divisor  
      
    \# A tiny epsilon ensures no NaNs occur if numerical instability pushes srgb \< \-0.055  
    clamped\_srgb \= mx.maximum(srgb, 0.0)   
    gamma\_path \= mx.power((clamped\_srgb \+ gamma\_offset) / gamma\_divisor, gamma\_power)

    \# Arithmetic mask generation: 1.0 if \> threshold, 0.0 otherwise.  
    \# In MLX, comparison operations cast to float32 create a numerical mask  
    \# without materializing divergent control flow paths in the Metal shader.  
    mask\_high \= (srgb \> threshold).astype(mx.float32)  
    mask\_low \= 1.0 \- mask\_high

    \# Algebraic blend: Multiply-Add operations are extremely fast on Apple Silicon.  
    linear\_output \= (linear\_path \* mask\_low) \+ (gamma\_path \* mask\_high)  
      
    return linear\_output

The evaluation criteria for this hypothesis dictate that the agent must run strict numerical parity tests. The output of branchless\_srgb\_to\_linear must match the baseline mx.where implementation to within a tolerance of 1e-6 using mx.allclose. If parity is maintained, the agent assesses the median execution time across 1000 iterations to quantify the reduction in pipeline stalls.

### **Hypothesis 4: Function Approximation via mx.maximum**

While algebraic blending solves thread divergence, it still calculates both the linear path and the gamma path for every pixel, executing floating-point power operations (mx.power) universally. A more aggressive optimization hypothesis involves utilizing hardware-native MAX instructions. The mx.maximum function translates directly to a highly optimized single-cycle instruction on Advanced RISC Machine (ARM) architectures.14

By carefully analyzing the mathematical boundaries of the sRGB transfer function, it becomes apparent that within the valid color domain of $x \\in $, the curve that represents the correct conversion path is always strictly greater than the incorrect conversion path. Below ![][image11], the linear toe ![][image12] yields a higher value than the extrapolated gamma curve. Above ![][image11], the gamma curve yields a higher value than the extrapolated linear line.

This geometric reality allows the complete elimination of logic masks, replacing them with a single spatial maximum operation.

Python

import mlx.core as mx

@mx.compile  
def branchless\_srgb\_max\_trick(srgb: mx.array) \-\> mx.array:  
    """  
    Hypothesis 4: Utilizing the hardware-native MAX instruction to resolve the   
    piecewise discontinuity. In the range , the valid sRGB conversion   
    path is mathematically greater than the invalid path at any given x.  
    """  
    linear\_path \= srgb / 12.92  
      
    \# Clamp to prevent NaN from negative bases in the power function  
    clamped\_srgb \= mx.maximum(srgb, 0.0)  
    gamma\_path \= mx.power((clamped\_srgb \+ 0.055) / 1.055, 2.4)  
      
    \# A single SIMD max instruction entirely replaces the conditional branch  
    \# and the multiplication masks from Hypothesis 3\.  
    return mx.maximum(linear\_path, gamma\_path)

The autonomous loop must benchmark branchless\_srgb\_max\_trick against both the baseline and Hypothesis 3\. Because mx.maximum removes two multiplication operations and an addition operation from the computation graph compared to algebraic blending, the hypothesis predicts a measurable decrease in Median Execution Time. If the numerical parity strictly matches the traditional method, this hypothesis yields superior Instruction-Level Parallelism (ILP).

### **Hypothesis 5: IEEE 754 Bitwise Sign Extraction**

To push the HPC optimization to its absolute theoretical limit, the agent must be programmed to explore low-level bit twiddling hacks.16 When operations involve checking if a value is greater than zero, the conditional check can be entirely bypassed by inspecting the sign bit of the IEEE 754 floating-point representation.

In MLX, tensors can be viewed as uninterpreted memory arrays. By viewing a 32-bit floating-point tensor as a 32-bit unsigned integer tensor (uint32), the agent can apply bitwise operations (mx.bitwise\_and, mx.bitwise\_or, mx.left\_shift) to directly manipulate the exponent and mantissa.13

The hypothesis posits that the sRGB threshold check (![][image13]) can be evaluated by subtracting the threshold, viewing the result as an integer, extracting the 31st bit (the sign bit), and arithmetically shifting it to create a pure hardware-level mask of 0x00000000 or 0xFFFFFFFF. This technique circumvents the Metal shader's floating-point comparison units entirely, relying exclusively on integer arithmetic logic units which are typically faster and more abundant.

Python

import mlx.core as mx

@mx.compile  
def bitwise\_srgb\_to\_linear(srgb: mx.array) \-\> mx.array:  
    """  
    Hypothesis 5: Extreme HPC bit-twiddling.  
    Evaluates the condition by extracting the IEEE 754 sign bit.  
    """  
    linear\_path \= srgb / 12.92  
    clamped\_srgb \= mx.maximum(srgb, 0.0)  
    gamma\_path \= mx.power((clamped\_srgb \+ 0.055) / 1.055, 2.4)  
      
    \# Subtract threshold. If srgb \< 0.04045, result is negative, sign bit is 1\.  
    shifted \= srgb \- 0.04045  
      
    \# View as uint32 to allow bitwise operations  
    bits \= shifted.view(mx.uint32)  
      
    \# Extract the sign bit (the 31st bit in a 32-bit float)  
    \# Right shifting by 31 isolates the sign bit (1 if negative, 0 if positive)  
    sign\_bit \= mx.right\_shift(bits, 31)  
      
    \# Create an integer mask:   
    \# If negative (srgb \< threshold), mask \= 1\. If positive, mask \= 0\.  
    is\_linear \= sign\_bit.astype(mx.float32)  
    is\_gamma \= 1.0 \- is\_linear  
      
    return (linear\_path \* is\_linear) \+ (gamma\_path \* is\_gamma)

The agent must validate this hypothesis cautiously. While bitwise operations are immensely powerful in C++ or raw Metal shading language, the overhead of reinterpreting views in the MLX Python abstraction may negate the ALU cycle savings. The autoresearch routine will track the graph compilation overhead and determine if the bitwise execution outpaces the mx.maximum approximation.

## **Memory-Level Parallelism (MLP) and JIT Graph Structuring**

The efficiency of deep learning execution on Apple Silicon heavily depends on Memory-Level Parallelism (MLP). While Instruction-Level Parallelism refers to the execution of multiple independent instructions simultaneously, MLP refers to the ability of the processor to overlap multiple independent memory fetch operations.20 If operations within the DecoderHead or CNNRefiner possess strict linear data dependencies (e.g., node ![][image14] node ![][image15] node ![][image16]), the processor must wait for ![][image17] to be fetched and computed before initiating the fetch for ![][image18]. This sequential bottleneck underutilizes the massive memory bandwidth of the M-series chips, which can reach up to ![][image19].9

MLX employs a lazy evaluation model.21 Operations invoked in Python do not execute immediately; instead, they construct a Directed Acyclic Graph (DAG) representing the mathematical topology of the network.22 The computation only triggers upon explicit evaluation, typically invoked via mx.eval(). When the mx.compile function transformation is applied, MLX attempts to fuse operations and optimize the graph, yielding smaller footprints and reducing overhead.8 However, the compiler cannot fundamentally alter the architectural dependencies of the neural network. If the user code forces a linear execution path, the compiler must respect it.

### **Hypothesis 6: Topological Reordering for Concurrent Dispatch**

The agent must analyze the forward pass of the CNNRefiner and identify independent sub-graphs. In standard PyTorch implementations ported to MLX, computing spatial attention weights and calculating upsampled feature maps are often written and executed sequentially.

The hypothesis dictates that if these operations are decoupled and explicitly programmed as parallel branches before their eventual multiplication or concatenation, the MLX JIT compiler will recognize the absence of dependencies. The compiler can then dispatch them to the Metal GPU asynchronously. This restructuring forces the hardware to initiate memory fetches for both the attention matrices and the feature tensors simultaneously, effectively hiding the memory latency of one operation behind the compute cycles of the other.24

The autoresearch loop should rewrite the forward or \_\_call\_\_ method of specific blocks within the architecture to expose this parallelism.

Python

import mlx.core as mx  
import mlx.nn as nn

class OptimizedCNNRefinerBlock(nn.Module):  
    def \_\_init\_\_(self, channels: int):  
        super().\_\_init\_\_()  
        self.spatial\_conv \= nn.Conv2d(channels, channels, kernel\_size=3, padding=1)  
        self.attention\_gate \= nn.Linear(channels, channels)  
          
    def \_\_call\_\_(self, x: mx.array) \-\> mx.array:  
        """  
        Hypothesis 6: Restructure for Memory-Level Parallelism.  
        By computing the spatial features and the attention weights on separate,  
        unlinked paths before combining them, the MLX computation graph is broadened.  
        """  
        \# Path A: Spatial Feature Extraction (Memory-bound convolution)  
        \# The Metal backend will initiate memory fetches for this operation.  
        spatial\_features \= self.spatial\_conv(x)  
        spatial\_features \= mx.maximum(spatial\_features, 0.0) \# ReLU  
          
        \# Path B: Attention Gate (Compute-bound Linear mapping)  
        \# Because this path does not depend on Path A's output, the MLX JIT compiler   
        \# can schedule these operations concurrently. Idle ALUs are utilized to   
        \# compute the attention weights while Path A waits for memory fetches.  
        pooled \= x.mean(axis=(1, 2), keepdims=True)  
        attention\_weights \= mx.sigmoid(self.attention\_gate(pooled))  
          
        \# Join Point: The Directed Acyclic Graph converges here.  
        \# Both preceding paths are fused and dispatched efficiently.  
        return spatial\_features \* attention\_weights

The validation of this hypothesis relies heavily on the autoresearch agent's ability to profile the Median Execution Time. A successful topological reordering will show a noticeable drop in execution time without altering the mathematical output of the network.

### **Hypothesis 7: Asynchronous Evaluation of Spatial Features**

Beyond topological restructuring of the DAG, MLX supports explicit asynchronous evaluation via the mx.async\_eval() function.23 For multi-stage refinement pipelines operating on high-resolution green screen footage, waiting for the entire deep graph to materialize before dispatching to the GPU can lead to intermittent hardware starvation. The CPU spends time building the graph while the GPU sits idle, followed by the GPU computing while the CPU waits.25

The agent should hypothesize the insertion of strategic mx.async\_eval() calls at the boundaries of macro-blocks within the DecoderHead and CNNRefiner. By evaluating the first half of the network asynchronously, the Python main thread returns immediately. The CPU can then continue constructing the computation graph for the second half of the network while the GPU is actively crunching the first half. This pipelines the graph construction overhead with the actual computation, maximizing hardware utilization.

Python

class PipelinedRefiner(nn.Module):  
    def \_\_init\_\_(self):  
        super().\_\_init\_\_()  
        self.stage\_1 \= RefinerStage()  
        self.stage\_2 \= RefinerStage()  
          
    def \_\_call\_\_(self, x: mx.array) \-\> mx.array:  
        """  
        Hypothesis 7: Pipeline graph construction and execution via async\_eval.  
        """  
        \# Build graph for Stage 1  
        stage\_1\_out \= self.stage\_1(x)  
          
        \# Dispatch Stage 1 to the GPU asynchronously.   
        \# This returns control to the Python thread immediately.  
        mx.async\_eval(stage\_1\_out)  
          
        \# The Python thread seamlessly builds the graph for Stage 2   
        \# simultaneously while the Metal GPU computes Stage 1\.  
        stage\_2\_out \= self.stage\_2(stage\_1\_out)  
          
        \# Final synchronous evaluation forces the thread to wait for all  
        \# previous asynchronous computations to complete before returning.  
        mx.eval(stage\_2\_out)  
        return stage\_2\_out

The autonomous loop must benchmark this asynchronous pipeline against a purely synchronous execution. However, a critical evaluation criterion must be enforced: Peak Active Memory. Holding multiple graph states and intermediate buffers in memory simultaneously can inflate the memory footprint. If the Peak Active Memory spikes unfavorably, violating the cache boundaries established in Hypothesis 1, the agent must reject Hypothesis 7 and rely strictly on the topological restructuring of Hypothesis 6\.

## **Tensor Layouts: NHWC vs. NCHW Cache Coherence**

The arrangement of tensor dimensions in physical memory profoundly impacts the performance of convolutions and matrix multiplications. In the PyTorch ecosystem, from which many computer vision models like timm and corridorkey originate, the default tensor layout is NCHW (Batch, Channels, Height, Width).26 In this format, spatial data is contiguous in memory. All values for a single channel across the entire image are grouped together, followed by all values for the next channel.

However, the Apple Neural Engine, the AMX coprocessor, and the Metal GPU implicitly favor the NHWC (Channels-Last) layout.28 In NHWC, the channels for a single specific pixel are contiguous in memory.

### **The Microarchitectural Justification for NHWC**

Convolutions on modern GPUs are internally executed as General Matrix Multiplications (GEMM).26 This process, often referred to as im2col, flattens the input image and the convolutional kernels into two large matrices.

1. **Implicit GEMM and Vectorization**: When data is arranged in NHWC, the receptive field of the convolution can be accessed linearly, and the resulting channel reductions map perfectly to the vectorized multiply-accumulate (MAC) units of the hardware.26  
2. **Avoiding Transpose Overhead**: If an NCHW tensor is passed to a backend that relies on NHWC for hardware acceleration, the framework will silently inject transposition operations (mx.transpose) to rearrange the data in memory before the convolution, and then transpose it back afterward to maintain API expectations.28 This continuous memory shuffling consumes massive bandwidth and stalls the pipeline.  
3. **MLX Native Preference**: MLX's native Conv2d explicitly expects the input shape to be NHWC (Batch, Height, Width, Channels).29 If corridorkey-mlx retains legacy PyTorch NCHW permutations to interface with ported weights, it actively subverts the framework's optimization passes.

The performance disparities between layouts are well documented in hardware performance guidelines, indicating that kernels not requiring a transpose operate significantly faster and consume less power.28

| Framework / Backend | Preferred Tensor Layout | Memory Access Pattern for Convolutions | Transposition Overhead |
| :---- | :---- | :---- | :---- |
| **PyTorch (Legacy)** | NCHW (Channels-First) | Strided channel access | High (if backend expects NHWC) |
| **NVIDIA Tensor Cores** | NHWC (Channels-Last) | Contiguous channel access | None |
| **Apple Metal / AMX** | NHWC (Channels-Last) | Contiguous channel access | None |
| **MLX (mx.nn.Conv2d)** | NHWC (Channels-Last) | Contiguous channel access | None |

Analysis of tensor layout compatibility and performance overhead across deep learning frameworks and hardware backends.26

### **Hypothesis 8: Strict NHWC Enforcement**

The autoresearch agent must systematically trace the execution graph and eliminate all dynamic shape permutations, such as mx.transpose(x, (0, 3, 1, 2)), from the backbone and the CNNRefiner. The hypothesis is that enforcing a strict, uninterrupted NHWC layout from the image input all the way to the final alpha matte output will dramatically lower the Median Execution Time by eliminating implicit memory re-allocations.

The agent must rewrite the convolutional layers to specify channels at the end. Furthermore, because operations like Max Pooling, Pixel Shuffle, or spatial interpolations are highly sensitive to dimension order, their algorithmic implementations must be updated to address the trailing channel dimension.

Python

import mlx.core as mx  
import mlx.nn as nn

class StrictNHWCConvBlock(nn.Module):  
    def \_\_init\_\_(self, in\_channels: int, out\_channels: int):  
        super().\_\_init\_\_()  
        \# MLX Conv2d expects inputs of shape (N, H, W, C) natively.  
        \# We define the convolution to operate directly on this format.  
        self.conv \= nn.Conv2d(  
            in\_channels=in\_channels,   
            out\_channels=out\_channels,   
            kernel\_size=3,   
            padding=1  
        )  
        \# LayerNorm initializes against the channel dimension  
        self.norm \= nn.LayerNorm(out\_channels)  
          
    def \_\_call\_\_(self, x: mx.array) \-\> mx.array:  
        """  
        Hypothesis 8: Strict NHWC Layout.  
        Input \`x\` is explicitly shaped (N, H, W, C).  
        No transpositions are performed before or after the convolution.  
        """  
        \# x enters as (N, H, W, C)  
        x \= self.conv(x) \# Output remains (N, H, W, C)  
        x \= self.norm(x)  
        x \= mx.maximum(x, 0.0) \# Activation  
        return x

### **Hypothesis 9: Fused LayerNorm in Channels-Last**

In addition to convolution efficiency, normalization layers behave entirely differently depending on the memory layout. LayerNorm normalizes the activations across the channel dimension.

In an NCHW layout, the channel values for a specific spatial location ![][image20] are separated by ![][image21] memory addresses. Calculating the mean and variance across these channels requires strided memory reads. This access pattern causes rapid cache misses because the hardware memory controllers attempt to load adjacent spatial pixels into the cache line, instead of the required channel data.27

Conversely, in an NHWC layout, the ![][image16] channels for a specific spatial location ![][image20] are perfectly contiguous in memory. Reading a single cache line fetches the exact data array needed to compute the mean and variance for that specific pixel. The agent should hypothesize that by strictly utilizing NHWC, the MLX mx.compile pass will be able to fully fuse the Conv2d output write operation and the LayerNorm memory read operation into a single, highly efficient kernel execution.32

The experimental mutation requires the agent to search the codebase for any instance of nn.GroupNorm or nn.BatchNorm2d (which traditionally favor NCHW semantics) and replace them with nn.LayerNorm applied to the trailing channel dimension of an NHWC tensor. The parity test must ensure the unmixing mathematics remain consistent, while the evaluation loop tracks the expected drop in memory bandwidth utilization and kernel launch overhead.

## **Agent Directives and Evaluation Criteria**

To successfully orchestrate these hardware-sympathetic hypotheses, the LLM-driven autoresearch loop must be configured with precise, invariant evaluation criteria within the program.md definition. If the agent operates without rigid boundaries, the mutations may introduce silent numerical failures or conflate the causes of performance shifts.

The methodology for the autonomous agent must enforce the isolation of variables. The agent must test one hypothesis at a time. If it attempts to change the tiling strategy to match L2 cache boundaries (Hypothesis 1\) while simultaneously restructuring the tensor layout to NHWC (Hypothesis 8), the root cause of any performance degradation or improvement will be obfuscated. The program.md must enforce a strict singular-mutation commit policy.

Furthermore, the agent must account for compilation warm-up. Because MLX utilizes Just-In-Time compilation, the first execution of a mutated graph will include the overhead of tracing the graph, generating the Metal shader code, and compiling it.8 The agent must be instructed to discard the first three evaluation timings of any new mutation to ensure it is measuring the pure Metal execution time of the cached graph, rather than the compilation latency.

Finally, the agent must enforce a strict parity tolerance. Hardware-level optimizations, particularly branchless mathematics (Hypothesis 4\) and bitwise reinterpretation (Hypothesis 5), can introduce minor floating-point discrepancies compared to the baseline PyTorch implementation. The agent's strict parity test must utilize mx.allclose(mutated\_out, baseline\_out, rtol=1e-4, atol=1e-5) to guarantee that the unmixing logic remains mathematically valid and visually identical, while tolerating the minor associative floating-point differences inherent to aggressive compiler fusion and altered operation orders.

By systematically applying these principles of Memory-Level Parallelism, cache-aligned data structures, branchless execution, and contiguous memory formatting, the autoresearch loop is equipped to autonomously evolve corridorkey-mlx into a highly optimized, Apple Silicon-native inference engine.

#### **Works cited**

1. Dive into MLX: Performance & Flexibility for Apple Silicon | by Pranay Saha | Medium, accessed March 11, 2026, [https://medium.com/@pranaysaha/dive-into-mlx-performance-flexibility-for-apple-silicon-651d79080c4c](https://medium.com/@pranaysaha/dive-into-mlx-performance-flexibility-for-apple-silicon-651d79080c4c)  
2. Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency \- arXiv.org, accessed March 11, 2026, [https://arxiv.org/html/2502.05317v1](https://arxiv.org/html/2502.05317v1)  
3. Modern Hardware \- Algorithmica, accessed March 11, 2026, [https://en.algorithmica.org/hpc/complexity/hardware/](https://en.algorithmica.org/hpc/complexity/hardware/)  
4. Algorithms for Modern Hardware \- Algorithmica, accessed March 11, 2026, [https://en.algorithmica.org/hpc/](https://en.algorithmica.org/hpc/)  
5. Apple M1 \- Wikipedia, accessed March 11, 2026, [https://en.wikipedia.org/wiki/Apple\_M1](https://en.wikipedia.org/wiki/Apple_M1)  
6. David Huang Tests Apple M4 Pro : r/hardware \- Reddit, accessed March 11, 2026, [https://www.reddit.com/r/hardware/comments/1gyh42k/david\_huang\_tests\_apple\_m4\_pro/](https://www.reddit.com/r/hardware/comments/1gyh42k/david_huang_tests_apple_m4_pro/)  
7. Apple M4 \- Wikipedia, accessed March 11, 2026, [https://en.wikipedia.org/wiki/Apple\_M4](https://en.wikipedia.org/wiki/Apple_M4)  
8. Compilation — MLX 0.31.0 documentation, accessed March 11, 2026, [https://ml-explore.github.io/mlx/build/html/usage/compile.html](https://ml-explore.github.io/mlx/build/html/usage/compile.html)  
9. MacBook Pro (16-inch, 2024\) \- Tech Specs \- Apple Support, accessed March 11, 2026, [https://support.apple.com/en-us/121554](https://support.apple.com/en-us/121554)  
10. sRGB transform library \- Project Nayuki, accessed March 11, 2026, [https://www.nayuki.io/page/srgb-transform-library](https://www.nayuki.io/page/srgb-transform-library)  
11. Optimizing conversion between sRGB and linear \- Excamera, accessed March 11, 2026, [https://excamera.com/sphinx/article-srgb.html](https://excamera.com/sphinx/article-srgb.html)  
12. Algorithms for Modern Hardware : r/programming \- Reddit, accessed March 11, 2026, [https://www.reddit.com/r/programming/comments/t8fmmb/algorithms\_for\_modern\_hardware/](https://www.reddit.com/r/programming/comments/t8fmmb/algorithms_for_modern_hardware/)  
13. Data Types — MLX 0.31.0 documentation, accessed March 11, 2026, [https://ml-explore.github.io/mlx/build/html/python/data\_types.html](https://ml-explore.github.io/mlx/build/html/python/data_types.html)  
14. mlx.core.maximum — MLX 0.30.6 documentation, accessed March 11, 2026, [https://ml-explore.github.io/mlx/build/html/python/\_autosummary/mlx.core.maximum.html](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.maximum.html)  
15. Question regarding piecewise definition of max function \- Mathematics Stack Exchange, accessed March 11, 2026, [https://math.stackexchange.com/questions/4893439/question-regarding-piecewise-definition-of-max-function](https://math.stackexchange.com/questions/4893439/question-regarding-piecewise-definition-of-max-function)  
16. Introduction to Low Level Bit Hacks \- catonmat.net, accessed March 11, 2026, [https://catonmat.net/low-level-bit-hacks](https://catonmat.net/low-level-bit-hacks)  
17. Bit Twiddling Hacks \- Stanford Graphics, accessed March 11, 2026, [https://graphics.stanford.edu/\~seander/bithacks.html](https://graphics.stanford.edu/~seander/bithacks.html)  
18. mlx.core.bitwise\_and — MLX 0.31.0 documentation, accessed March 11, 2026, [https://ml-explore.github.io/mlx/build/html/python/\_autosummary/mlx.core.bitwise\_and.html](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.bitwise_and.html)  
19. ml-io/doc/python/tensor.md at master · awslabs/ml-io \- GitHub, accessed March 11, 2026, [https://github.com/awslabs/ml-io/blob/master/doc/python/tensor.md](https://github.com/awslabs/ml-io/blob/master/doc/python/tensor.md)  
20. \[2201.00485\] Freeway to Memory Level Parallelism in Slice-Out-of-Order Cores \- arXiv.org, accessed March 11, 2026, [https://arxiv.org/abs/2201.00485](https://arxiv.org/abs/2201.00485)  
21. MLX Quickstart \- Daniel Liden, accessed March 11, 2026, [https://www.danliden.com/notes/20240401-mlx-quickstart.html](https://www.danliden.com/notes/20240401-mlx-quickstart.html)  
22. mlx-dev | Skills Marketplace \- LobeHub, accessed March 11, 2026, [https://lobehub.com/skills/luqmannurhakimbazman-ashford-mlx-dev](https://lobehub.com/skills/luqmannurhakimbazman-ashford-mlx-dev)  
23. Writing Fast MLX \- GitHub Gist, accessed March 11, 2026, [https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)  
24. Part 4.3: Transformers with Tensor Parallelism \- the UvA Deep Learning Tutorials\!, accessed March 11, 2026, [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/scaling/JAX/tensor\_parallel\_transformer.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_transformer.html)  
25. The Execution Process of a Tensor in a Deep Learning Framework | by OneFlow \- Medium, accessed March 11, 2026, [https://medium.com/codex/the-execution-process-of-a-tensor-in-a-deep-learning-framework-a4d853645d5b](https://medium.com/codex/the-execution-process-of-a-tensor-in-a-deep-learning-framework-a4d853645d5b)  
26. NHWC vs NCHW : A memory access perspective on GPUs | by Deepika \- Medium, accessed March 11, 2026, [https://medium.com/@deepika\_writes/nhwc-vs-nchw-a-memory-access-perspective-on-gpus-4e79bd3b1b54](https://medium.com/@deepika_writes/nhwc-vs-nchw-a-memory-access-perspective-on-gpus-4e79bd3b1b54)  
27. Why does pytorch prefer using NCHW?, accessed March 11, 2026, [https://discuss.pytorch.org/t/why-does-pytorch-prefer-using-nchw/83637](https://discuss.pytorch.org/t/why-does-pytorch-prefer-using-nchw/83637)  
28. Convolutional Layers User's Guide \- NVIDIA Docs, accessed March 11, 2026, [https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)  
29. mlx.nn.Conv2d — MLX 0.30.6 documentation, accessed March 11, 2026, [https://ml-explore.github.io/mlx/build/html/python/nn/\_autosummary/mlx.nn.Conv2d.html](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html)  
30. Difference in weight's shape between Conv2d and PyTorch's Conv2d · ml-explore mlx · Discussion \#724 \- GitHub, accessed March 11, 2026, [https://github.com/ml-explore/mlx/discussions/724](https://github.com/ml-explore/mlx/discussions/724)  
31. Analyzing the Impact of Kernel Fusion on GPU Tensor Operation Performance: A Systematic Performance Study \- MDPI, accessed March 11, 2026, [https://www.mdpi.com/2079-9292/15/5/1034](https://www.mdpi.com/2079-9292/15/5/1034)  
32. Faster Models with Graph Fusion: How Deep Learning Frameworks Optimize Your Computation | Practical ML, accessed March 11, 2026, [https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAXCAYAAACYuRhEAAAD00lEQVR4Xu2XV4hVVxSGlxXF2LCgsYslgg2swTaK3RcF9cFE7Io1D3mwGxUU7OVBfFAcFBGUCAo2bGPHBoIaIiqCmKCCJcaG2P5/1j3DOmvOOfcOyIBwPvjQu/a+d/ZZu60jkpKSkpKSjSbwN9gD/gCbwdFwrO1kKAsnwAa+IUN1OB2ugXNg/XBzqZAnOsYOsBJsDyfBPqaPpQr83QcNveCyjENdWxF94RfnP7C77QSmwH3wmWifLuHmQlrBK3A2nApvwddwuO1UCvCB/TOdl/Ck1oTz4An4Hr4xbZa1cBf8BW6En0S/wwkKkQcfiSbgIFwI69kOGZicYXCdxCeyAI4zn+vAj6KD5P9Li6Wik3gV7oYzYAXbAdQVfSau0rMSncg8eANWM7FNos+/2sQK6Q3zfTCB+RKdSG75D/Az/NHET4n2n2xiUfDBuMXiKAMb+2AMS+B4H0zgmEQncpHo2LeaGPPFGBdfCO7/fB9MIC6RZAs8BMu7GPtPNLEousLjEp79AE5SPvzVxeNYLN8mkXzGy3CEiXHC+TwPTayQnvAA3AxPwuuiyY0jKZFRcHtxe+eytfuLHg81TIxJ3AmnmVg2uJJWiZ7p5+Bh2CjUI0xcIqPg8cbnZ75C/Ayfw86Zz63hU9GzMoqSJHKAaN9tviEBfodnFpPJJPKgZxVQEjhGLojamc8j4Us4uKhHmJIk8gJ8Bdv4hsqiJZBlh+gqauHiJEgkt2ISVeHfcD+s6NqyMVA0mbwoZrq2XOBlaVc1eQDvi06Oh4l864MRsBL5H/bzDXGskPhzLUhkN99g4Bl5BG6H5VxbLvD7vKQ4ESxTSgovJg9XEsfd3DeIJvKdDzpYJv4rCTuR9dU1Cc/UctE/OsvEAnJJJBPIMyqgnehAcoFJ3Cu6nfvAM1J8dSXBiuEJ3ODiXOEcd1sXJ9kSyfHfhD+ZGF9iiuDMsfjmzNubltc9/yjfdjxBIn3BHsBieK6LLRA9p7JhkxjAZJ4WfWPKBZ7xUefyHdHt6+tJwkSyKI+ClxR3h72s+Bu8zUOwsLSvPZx9Xu388Sj+EB0ozzFPcIbwwfnHC+BF+J/orCbBI2CP6BuUh7Ubfy/XZB6FDc3nTqJj5iKIguNlDcxXZAtzcVu0uGdFw35c2X/BP02/Qngws1O+aJK4hFkusF6ycHD3RG8/JoYJuyta/BJeWrygOGAvX6uKvVI5hkj4rcjD3ZHrxcPy7RJcL7pQHmf+tbuuluj4H4jewvSF6O4clOkT3BVRrsz0KUZLOAZ29A3fKTy2WNqNgk3DTSkpKSkpKd8ZXwFe9dz/HROyTQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAXCAYAAABZPlLoAAADeElEQVR4Xu2XWahOURTHl3mMzFPqKpmJSOHhhmSMErfEA4XMFA9ePIm8mIco6iLyIBIpRYpCPBCRTNc8JHPGTP//XXv79lnfud937qG8nF/96/vWOmcPa++11z4iGRkZGf+XvtAKaBLU0PgOQlOgAVAfqJdTT6hj8FwhGkFLrTEGtvsAeg49he5CLSNPRBkEPYSeOVVAtaBy0ffZDvUYugfdhu5AJ6AZUoR60GnRxoZDa6Cjgb8+9AP6VYW25x7Noxm0HDoJfYE+Rt0FGQ/dF+1jcdQVYRd0Q/S5TsZXE7rifCWBvYHoPGlfH9jz2CPagYeBYjA4McKVZNT5zDZoM7QR2i+6Mq3cc3G0hhZCpdAZqV5whkJboBeiE4yjCbRPdMycKPuznJOqfbdEfZOtg8wWdXYPbONE08szEVoW/Cc1RHfXMGMvBLdxdYOzDlorOsZ+UXclM6EJkiw4cal5SNS3yDrIYeit+91U4hvgzikxNja2ytiKkTY47J8T2BR1V8Lx15FkwWlh7DxO3oiOyWdJhEfQTdGJMr0uizYW14mnG3QVqm0dRUgbHHIJegXVzbmli+T8SYLTWTQI3AAsKheg89CQ3KM52BBf+g7NcjamC7faNdHDLA6m0wJrTMDfBGee5J8Nq6He7neS4ByAdotuAp6XrFysVqyiebAM86WvoqXPM8fZxwY2Tw/oJ9TBOhLA4HyyxgIwOL6ScMVZ7Y67/1y4I+43SRIcm1a8gjBTXrvfEVhl+BJP7JBpzr7V2MkO0TxNA4Pz2RoLwOBsCP5z5bnL20MjJbp70wSH8ECn75h1MPrvRc+PkKmiL8TdX7gVmf9pSBMcXhk8DAjHxXtTOdQ88CUJTlyxYVrSR9mLb+Vq8A4T4st7mbG3c/ZTxh7SX7R6xMHgMDWSwuCEFYqLyQLCBeLZEZI2OPOlwJxKRS98AwMbL1U8yXk4h4wQbSi8PYdMF/XvNHYPJ/ANamwdVcBPDb7D26yHVZV9jAps5KyzM+UsF0V9bYx9NPRSdMG6Gt8fOAiuxkrRThhplmsLSx472WsdjsGiKzs3sDHP+S1TIZrCFM8sXh+YJnHwXsNvoQ+iz/MetsT5WI6vS66SMvX57eXbfiLan/+24u2aY6beiWYJx0I7A8MLZokUoa1oWRsj+TvGQztvz3FbNyMjIyMj49/zG5lW7ox1OEEIAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAXCAYAAABefIz9AAADE0lEQVR4Xu2WWahPURjF1zUPZR6K6GbKkOKBpMxEKEQuCfdJlMjwZgohwwMyJZIhUoa8eDJn9kAZMhTXmDFCZIq1fPvc/z77/o//6T55OKt+nXO+tc8+5zt77+9sIFOmTP+LishIsopMJK3idrn6kKWO4YGXTzvIC/KSPCdL4nYFHUSu/TMyl0wnT11MyH9EHjiukgWkHhLUgpwgC0lfsp18JRP8RtQ6spdMIuvJL9h9tfxGeVSFXCNl5BWpHnNzaktukN9kd+BJG2FeaRCfQj6S86Ru4P2VXvYt6eWua5Af5D2p7WL9YQ/3v9IG2APXeLEknSWLYe1HB16k5WQ2rM2WwJNWwrzxoQEbFHmbQ0NaCzMnu2uNyHfymdRxMY2u2mxz15JGWzFNpUJSgm3IN3Is8CSNsuL9UDjBcaFBzYJ5R0JDUuftvevesMYHvFgPcoWM8WLNYO2eeLEkKUGN/iHY7GgetzEENnppEhwbGtRxmDc0NEIp0euwZIrjVgWNgHWqtVFIUYLRPfPjNnaRJkiX4FTSkDQmrclO2EdW/J/aSm6T16Rr4OXTBdji7hQaeRQlWBVWBW95Xn2yz52nSfAMrAjtgd13jyyDFctUGkw+kUWh4WkarM3A0EiQElQi0mrYi/Z01+pLvygpTYLhFFWdUHX/SUYFXqIuwjrrHBrUANgoaF2mlRJs4M47wvrWjJGOkmruvDIJSu1gnv610Ycslxb3zCCmKaAb9OP3pal7E/aSkXR/ISlBrZtIl2C/oe6wKh4pTYL5qqj0DubHNiDFLij8OXzaxWZ4Me1uTrljJP20VZAKSQk28q41LdX/HcRnSWUT7ALz3iD+HNQkj8l+L6bdgL6GaOliml4qQCoOJ2Ef4BzsBQ+7NklqSh6SYV5MBecLuezFpEGwF9WPO5Q2FPJKgnhU+eWVxi1TB3KXbCLzYNNHlamb12YFciMdoi+bJJVwVVqhoqQ1Ekn71FJ3rhG4D6vgavsBts+cg9xeVEVEz9M2Uv2UuaPaaw8b7cTyqgi2vrTR1o4jU6ZMmTIV0h+iRtRvam8JxQAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAXCAYAAAC4VUe5AAADC0lEQVR4Xu2WWcgOcRjFj71soSRSXFhL5MaSuFFKyppsUYSylVzIUkKWrCl8lOKTLClLcuFCSdlucGNfP9mXhEhCnOOZ+d7/PO+8ZvpyZ079eud9zn+WM/PfgEKFCv0PakJOk07eqKBWZCrZQGbAzv+bZpKn5BV5Qc4k7TItJM/JS1j7o6QveQK7RkwNeUDuk5tkG+mKnNLD/yI9vZGijuQW2UIGkt3kNmkTNqqgrbAH1b36JK1a1SOnYG0ekgZJG+Mir9rV9Sx3YS+ju/PK1I98Rv7QR2C9ItRFss/V0rSKrIDdS18lTUNhX1pt9HK9hsG8Km9Qk2GevnpFqVueJ5uRL3Rb8o0scvU15Auyu7lCjyRXyVvSOGn/0V7YfbJC7/AG1RvmvYP1mFSthb2dxcgXejSs3XRX10tQfbCre8Wh58Paj03aaEkORsdZobd7A6Uc670RSxPDseg4b+hZsHaTXH1eVJ/g6l5xaI1/9Rg/THR9+VJW6D2kNexa7WBD4j1ZTZrVtg7UkJwjHaL/eUMvRXq4OVFdv39THFrSjPyDtC/ZOAF7Nikr9B2yP+IAuQS7ZqUJEsuR7KJ5Qy+BtZvo6nHo2a7updCjouPhsHN0b0kzruaWWFmh07r3Api3yRsKpiUhVN7QWm/VboqrVxqjXgqteUHSUqS1WMudpHHYKzqW6hJa0n7gJ+kfFufCFvka8ijiI+xCOuFKbctyxV/Hd+NlUT1xoxQp9Jjgv4LqvEHkZFCXskKnzd6S5in5G73hpbeW9qW1dHQJ/jeFLU1+dtQG5Q3KNxJeCh32hm4ohdPHCFWX0PXJa5g/xHll0kOroZ8EHpPvpHNQqyLXYDeQtNZqK7gyblBBjchxsg6lcyVtbL7CtraxNJnpee4FtVgjYN4uV29BDkdeddJKSmu0tnofyCfYlH858A+RG6R5UFPIsxHq5hfIzsBPk+YCXVv3EOoVAyJvGmxDEus6eYZSWw0/7QLjvbdekIJp5tfeXDWhtnqB45Hd4+qsHrD12g+JQoUKFSr0r/UbZELYYbUYdZEAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAXCAYAAABZPlLoAAAEBElEQVR4Xu2XWchWVRSGV2aWZVKmTVgZRmqDERVhUdJFM1QGFXll1kWlZlFCZlBESUSjjRdFAxRFkAhmREVBkKYXmWiDlVrZPGgTVDS9j2tv/71X5z//hzfenAde+PZa++x9ztrDWp9ZR0dHx/ZhB+kc6Q7pWmmP2l0xXLpYukE6LPjaOF6aJ90oHR18kSOkT6WvpS+lT6SRVY+aSdJn0ldJ66UdpSfMn2cctFFaJ30kfSy9LF1iLQyVXkk6XbrGfIKxZacEAWRiBjzV/KWPrXo0c520RjrfPLBrpZlVj2aYb4P0rzS7dlU8Jr1n3u/g4BskrUy+MYWd77492e8p7BUPSL9avVuelpYXbThK+tH6gjbZBhg4McG8XxnE06Q/pEMLWxOnmL/fN+Yf2AQ7mfd93XyevWv3Ft6y/n0sFL4LomMn6U/p3WCfZf7AuNQebL5tn9vaw2x382M40BG5W/o52IZI/5gfsTYIDs/fZf4+TXNdJp1rvQWn6Wi+YO67Kjr4eBxvB/u0ZJ+R2uwa2leb308HmZ/pXmDFuT8iBOy1aAzk4HAHMf+C2r2FheaL3Etw9gr2naVN0m/SnsFn+5k/FHcOQcF+a2pfmtq3SEukx6VvpSuTv40vpA+jUXwnvR+NgRwcWCH9YL7rMhzL7O8lOIeYB4EdNFFaJi2VTuzrWsOlGlf2EfPBHk7th1Kb80m0Ie+mC1O7P7hbmoLAPYLaKIPDQjBfeTfMl45Mv3sJzrPSk9JT0jPmmYvksltf15qzpb/Nsw8caB4wBmNyeD614x1BemRF2/hd+iAazQNDmm2D4OQLnxVnLHYukIUWpd/QS3DisTpAesc80fC7kbPMP5K8/6g0x3ywy5P/wdSemtoZagXso4K9hNqCuiLyvbQqGgME596izcr/Je1vXnaU5cC2BAe40PEtjo7+uMn8gbxlqVNoU6eU8NHY9w32Ei77eHy41DluLwV7hODcV7QJCPNdb17gjSh8vQSnKVvxjfjQrsFnF5lHjdSceUN6s2iPN3+YKJdwLCju+NjMMebZI3OzedrepbARTMb7X/oMEJwyQ3GUPje/K7g7SrY1ODn5NGbO+83vHAIAU6SfpOO29nBelV4s2mQKBj2zsE1LNo5mhqLxF+m8wjbd/L5q23HAXxk+mmo2c5v5HGcUNmAxsXPkIhS0+PYJdt6drMldlmu6CoycfS5bshMp+qSqh8OHrDavK+403zVzqx5mJ5iv7BXBzkvwl4RSgJKdC//wqkcNdQ3BI6jUQ5vNaywgHbNb2UVAZmU8+iFKB457/m/Fkc7HhkXnDlyf7ASGAnOMtcDK8AEnW3txx/Ehq7Hyo4NvIIaZX/zMUx7hjo6Ojo6O/vkPB+oIZdDCy4YAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAXCAYAAABefIz9AAADO0lEQVR4Xu2WWajNURTGlykhMxkSZY4khbzgQTJmjnjgGJLIzIMht5AylDInkulBRAolHq4o04MhlPmap8wiU3zfWXufu/e65557uk/K/6tf96z17bPPXXtY/79IokSJ/hVVAEPAGjAf1IntSLXAWLAYdDCe1Q7wHLwEz0BBbJfQASke/xTMA9PAE5cj9B+Cu45LYIno/5VV1cApRz8wF7wArcJBTlwETj4R9AX3QddoRElVBJdBEXgFqkRusfh7V8EfsNt41AZRL2Xy48EncA7UMF5am8AXiXdtv+jKhOoM3klx4b1Ff3B9ZkTpOgOWiY4fZjyvFWC26JgtxqNWiXqjrQFtF/U2W4Or+QNcM/mZol9o5+LK4LHoEfKqKXqkuwS50sQCW4Lv4KjxKO4y837RchU4yhrQLFHvsDVYAI2LJp9y+Rku5u4xniN6X1uASs7LRyyQd+QQ+AkaxXb6uHP38ilwpDWgE6Ier1ikJs6wO8jCmF/p4skuXi462S7wGkx3flnyBQ4SnWdBbKfnayD5FTgB1AX1QXOwU/R0MZ9VbBSPTG6b6GRbXcwfZHwHVHU5v6vZ7oSVL5C7zi54I/Bqg33ucz4FFoo2oT2i37stuvBNMyONuKq/RY8JxVVh0ZyMk1IHXbzUxV5s2+yQZYkFshBqtehc3V08FQx2n/Mp0B7R6mAv+AWGGi+jgaL/6EnRZ9dC0cn4DKLYnRiPc7HXPZdvaPJWLNB36fYSn44jok2MKk+BVGtRj89av5A5VSD6hU4u5p1hPCIzQsUHLfONTd6KBfLeeJ0H70U78Nogn0+B2boo9VbU52ZFGgOOibZ9r0JwNoj9qk8JchTv003RzppLLLBeEPNYcr5bEr8NlbfAjqLeG4l/J62NoneQRVDDwUfQLTNCdRocD+K2opMOCHLZxOP7APQPcmw4X8GFIEf1EZ2TD24rPnPpcUNCtQFXnJeKLRWfhddFGwjvBdt/z2iEiseQ3Y93Zp3o7i2KRpQUWzhfo8hn0Tvixbuecp+5A+zQ/G2O/SB6/Pna6N9F2URYxDfReYrcX47nC0gPySG+j3IneknuBziPIrvtJNDMeIkSJUr0f+kv+CnbhtMPnhkAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABZklEQVR4Xu2UvSuGURjGbxJJmSwsL+UdTLIo/4UyKGWQLFIGZWE1KJvFbjJYfMYiDPwBJh+vzyQk35Piupy79z3P7bznsRjUc9Wvnvu6zzn3ec5z7kck039RDVgBOY2rwRE4Bof6TNY0T3WBcx3D3Clo9vI/NA0+QZvxW9S/E7cRqwVwAEZAo8kl1AleJVykQf1L41PdYAvU24QVd7cNZiRe5ML4PWAD1Bk/qCnQB8bl90V6wSqo9byy6gCL+pxW5ErjfvABxoojIqoSd55NGqcVuQaDYBk8gXeQ98YFNQEGvDitCHc/L25zQ+rtgsrS0KS40JLx0orcSHLBTfVHPS+hYXFNdAYKCo+Ak3hV94sjwx+eyoEX8AZaTa6sZiX+JqE+YQMytyORY/M1J25Cu/HZxfRvQYXJMeYbMj8pkULskRPwCJ7BA9iT0r+Lt4o+4TVed9O+xf8Yj4s5Ht29uPUyZfpDfQFc+GJIm+orCwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAXCAYAAAC8oJeEAAADq0lEQVR4Xu2WWQhNURSGl3kMKRkzExHxYCZEMss8hMhMpmRMKVOKiEyZhcjwIMo8ZHyQF0KZpwdCxiTE/1t7Z511zxXP7l9f3fOfffY5a9+119oiOeWU0/+qvKA7WAymgfLJ239UKTAULAMjQaHk7VRxbG9vQtXBJLAEdHX30sR3HQFVnM/v6Ss6XzHQEkwFle0gqiA4AI6CNmAmeA6a20FZVAncAitEx28At0FpO8ipM/gBxju/HXgMpoD24AzYlxiRKS4i56rr/PPBt6wXjTWhceAFKGo8ZsBdkN94adoruvJWl8A250WVBPckM/h84BmYYTwu4Acw3HhWTcBHSQ/+HLgMrooG3TNx1+g6OOy8DqKTtnK+VRnwBUx3/iLwSdLTfxOYI5nB9wheY+NRF8Ep51Gcm//uckkPnllT1XkZ4n7lwzucz4+gP9/5Vr1Ex4xwPheDfmvndwQbRf8xH/yq4FUzHsU/hQvsM5CZOVh0i6YFf1r+Ivhaog/zo6zqBZ8pk02jRccMcv7E4A8wXnFwAZSQ9OB3B88X2v3BL2u8RuBg+J0teGbLBHAcXAOrQZHECNEilRYkJ6P/p4IT09cGSTEoH9wa0CX8TgueH+mDpPh+Gxwz4CyoEK6zBc/5toACos+sFC3EVc0YaSb6MKu0VQx+j/OtZouOGej8GPyYcM303/b7dmrwx4JXznhUDL52uJ4nyW2WLXiOZ/uOqiI6brvxpGYwWYis6gef6ZJNo0THDHE++zR99vHCooXJtr604HcFr6LxKLZg+nyeAfrCnC34PO6aGcBxT6zJvfhNMtO7hejgWc636iyZQVBzg99UdH+yd5OH4IFoW+X9V+GaRW5Z8HwQJ0Q7B4PhHuY8j0SfI+9En3sq2taobuAt6BeuKbbS+M6EzoErzmMqc3AD47G1MVOieC7ghy01HsUt9FL0hWnqI5mL1jZ4nYxH3QGHnGfFWuIXbSz4DoYZj4WU404a75dYsBhELCLUTtFFseI/91WSRWOd6Dkh7i+eoHiIWRAHpCguLE9yUVyoG6ItL4r79rPomSObuNCcq6HxuEXY6pjqUWy/XJDUc8tC0WrID+L+4+mIbcmKxe+m6FaJYrBsK4T/JA8la819qxqip8bX4L1oysa0p3jufgS2in4sF9G30Sj2+Pui6c253kgyeyeL/nkLwGbRd/ralBArbX/R9mdX7W9UR/RD/Z79V7Et8f3cr77n/6tYbLn/WZtswc0pp5xy+r/0E3rS6bvA4908AAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAYCAYAAACSuF9OAAAB7UlEQVR4Xu2WzytlYRjHnzFNiSE2SGxmI/IjkUHSKaKw0ZSVJiJsSCQmCRsLSmJhoTCKhUg2MjWNBbtZ8E9gwUaJxDS+T89zzn3POV33Sp27cD/1qec+73tOz31/HqI474zP8AcchcVwGFa5egQIFzMDP8E1uAN74YbZKUhyYZbGJ7BZc2lOjxiRDO9gqrchaOrgNGyBZ5qrgI1Oj4BpI1k7s/APHIAjrh4xgKfpA0zQOE6g/I+hcd5MJuzxJkEJnIQNJFdGOHLgBOwmdz8+RDvhIMmpHhWL8BCee/LfYD8sgL/glrvZga8Qfp6PgQ64rHk+HlZILuHv8J7kXVFhkb+g33BB4xqSxcgv93IKhzROgrck5xQ/80Ay+gy/b1fjiFjkL6gWFmpcRFLQl1CzwxXsM37/g3kkXwftJCPFcDE/7U6RsMhfkMk83PcmFf7nYxrzOuHCebpNsuE1veLbyYIX3qTCI3UAE70NCo/iMcmlOw6fSEbU5iPcg01GLiIWvPQmQT5cJSmG/324nZICv8IMkim0p4mZg/UaVxv5F7HIXxAP8zYsh2UktzyvDYa3t71jWrXNjs3dyFPJ66sUVsIloy0s/I3DU3ID10mGnjki93H/SKEz5i/s0pj7b5KcY7ye0jVvkf/KmNI2h2f0xW+zu86vnwAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,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>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAYCAYAAAB5j+RNAAABTElEQVR4Xu2VvStFcRjHH0TylpeUpAxGi4mklJHFYJBkMllksNi9DAZZDFaDwcJiVQYZ5C8QmRRG8hLxeTzn5tync526zu1cOZ/61D3P91fne37ndH8iGf+YFlzyw3JhC9+w0gdp04fv+IHtLkudI7wQK9frslSZwk3cFis3kh8nwiAO+GEc9XiOzbgsVm4yb0Uy9OCOH8axgrPB7wWxcvPfcaIc4pAfFkKf5hgrgusZsXK6g6WgFfewzQdR7GN/6HpMrJx+e3FUY2cRDuMVTssPjOILXostVm/Eyh2E1hVCH2q3SB/xErslgho8ww437xIrd+rmSbKG61jngxx6RC36IdSKldNdLAVzuOGHOfRYmsAHbHSZot/RU2DSR1gVnkj0fb/+BO/EbvyKt9gUyvVcvQ/yZ7G1q6H8t4xL9NsqC3THGvwwIyPjL/EJOrNCKIDGStAAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAABbElEQVR4Xu2VSytFURiGX5JIBu6XMmEkA5civ8GEgRhhioyMKEO/QclAKBRlYnpSBsYYGCi3CUqiXEbE+/WtOnytfXTs3XHSeuqZfO/u9K6911kLCATyix56Rm/oLb2ml/ScXtEjOkXL3PN/xhr9oO1mPujmKTPPOafQN+jjGFqy1Qa5oh5aYNMGpIg+0ndabbI4VNAxO4xiCFpw3AZkDppN2yAmBXSPFtvAxwK0RC90ZVW0iy7REzqafjRRZuisHfqQEg/QP8qKc5te0AFamH40UeQtLtMOG3ylDtH7rxOabdggggbamKXN9JDO0xJ4GIaWmLCBQ97uG22ygaGcrkMXk63ypZ5oHzwsQgu22cAhR4/kPxX8Lf10Bxl+X86/O+h+sIxAy+3aICHkUjhAhj3eAi2wZealdJK+0n1a+T1OjFXabYeCHCFy1z5DC75A712ZiffQq01KRq4uJjXQT5u3yMJr7TAQCPwHPgHusVCe5T/WegAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABAUlEQVR4Xu3SsUsCcRjG8bcol6LF0MQhqlFaGhJxc23Owck1CFoiWqIlcJa2hgbdGl2iP0AUodpram4INGhTvz/eQ15frnKU6IEP3L3PvXcHdyJ/Mgns4gAZM1/DhjmfSgrX+EQPN2ihjjS62J9cbXKCPppYd90ZBpEl18kxRjjyRZTwqkM8+KIcFbe+cHnGuR2s4F30qVlbxCQ8NW8Hp6KLj3b4TbaxYAcd0eVLO5w1H6LLe774Lcuii1++iMkVNv3wTfQGq74w2UHbD0NqosvhN4zLIu5Q9EVIEq94Qcl14ce4R9XNp7Il+g3DGzzhAg3RxYK57sfkcIhKdPyfucgYas0sUY2rlVwAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA0klEQVR4XmNgGNZAEIgr0QWJBdOA+A8QM6FLEAIGQPwXiP8DsSiaHEGwH4jvMEA0a6PJ4QWRQDwRiGcxQDQ7okrjBtxAfA6IBYC4hQGiORxFBR7QCsTJUHYBA0RzHkIaN1AG4kNAzAjlxzJANINcQBBsAGIzJL4XA0QzyO84ARsQnwFiCTRxGQaI5hNo4igAlIpK0AWBgIMBovk+ugQIgFJOMBB/AWJeNDkQYAXi71CMksosgfg1VOIXEL8CYj4keVDSfAOV/8EAUduGJD8KBj8AAGTSJtpF0+ZhAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAABB0lEQVR4Xu2SPy9DYRSHfxWLmJSEpVMnMTQM4mMYhKlmpFNnn0MiJjGQMFotvgFDB4k/XZCI1ICpDc/JeW9bp73dJX2SZzm/3z25976vNCayivf4gq/4jE/4gE28wRpOp34uJ/iDlTDfSPOrMB/gTv4Gw7iVL1mMQcaCvHAWA5jED+zgXMi6bMoX7MQA9uVZPQb9HMhLaziDs7iCR9jA7V51OFZqyX/kcfICH3EdJ3rVQeaV//3L8uw0Bv1syUu7MUjY27WxFIOMQ/mCpRgk7Ggtz11g5/+GhRhAVf7wZQwyyvLCeZhP4R5+4zUW/8Z+RHbXP+ULvuT33mbmu/zq2pKRJzDm3/ILy107yeB3aPYAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAXCAYAAACcTMh5AAAEjElEQVR4Xu2YV4gkVRSGf3PEnFBxERTjg4iKedcAYn4w51FMYEJFMCBiDigYMKDotGsWcwLBrJgwgL6Yd81iRkUR4//NqZq5fbq6ph0EFfuDn5k+59Z01bn3hBppyJAh/z4WyYb/KutaO1prFrY5rNWLz383m1vnZONkTLOOsTaxFrZWtna39isXNbCHdVY2FqxjnWrtai2YfP0gQMdZ71rvWLdbo9Z11hrWCdYV46ulw60Prc8qfWLNst6u9KJ1igY/Vdcq7vsvsYX1R9LH1oblosTy1jfWbdlh5rMetzrWVtZ51v3lgj4sZN1rfW8dnHxrWW9Yv6g7gDWXKu57JNn3t76znlH8/TYWsJ7NxkGYYX2k2K37FDu2XLmggTut39QcwJmKnawhmKxdvLA1cZUiCJyqJsiU39UcQNKOa8mczNUK3+XZkdjLOjEbB4G872RjC/taZ1g/qjeAhypulnSr2UGRym1srAjO69ZcyVfytNoDSLnIHK3w3ZUdiQetlbJxEDbT4AFc1nrSmlfNAbzb+rb6fVFrqcLXxmWKh7woOxJHqT2Au2SHeUjh2yY7Csi4R7JRUZOnW0coNnlbRYnqYlNF7aGOPGq9rAhqEwRs/er3pgBS0KlVZytS+VVFXVmmXNTAm4qHPDI7EjSjFbJREwE8QFEqllScJkrJB5W9jWOtg5JtCUVGcC21/BrFd/Qcio2sr631qs+rWZ8ramEJ6XF+8TkHkCDxBb9ah1Q2dpDU4UbmrGxN0JC4dqfsGJA6gE9Y1ys270bFxlBuaHptsMm5UzOZ3FN85llmqSGAdJ9pyTaqCMQq1Wd2g9Sdf3xFbwDplDzEz+quYzQF7NsXtgwjC2vyKYAVFffHT04fJ4tRq6RfCnNib1A8y87JV7O2ejMJqNvU5YutLRVlixm0rUaPQwqWD9RRzIklOYBLK655q7ABTQd7Wxe8UrGG780QnIetTxVrfrBu7VrRP4DAIcDHaEZdzpBVNLoMpeApxbWI8YpU74EZ6SV1pxjHnovqmsRQ+r7iCL9nza78PAyfGQG4npnrtbhknH0UaxlT+sEOs+a57CgYUaxhLMm0dWH4SuHfLtm551esuZMdOHH4N7BOUjwXf2NGsWYsr9kZCn/5R+qZLJ+6Goo0/nz0ORnMlCX1aNM0o5XcrFi3Z3ZUHKbwc1ozbQGsS8sXilJUwsZdkmw1NCCaRw3x4QCdXtjGuEDdO7OYonORNv2g7XNTjC0l0xVDM7tWc5P1vGKz2uDhyAYaym7Jx80/pvjOpjGGZ8DH62XJqopJAN9It2uMUU00z0zHekATmcn9U6t7NolgML50rNMUHZPZqd/oQZrx3km6IoJNCtccr0jrMxWDLx1u0H8AzKN49ftS8T3MhxcqRivqIwE6d3z1xLswTYIg/aTIqNnVT+6PLGl6LaV5vpCNBR3rDkWQmU9vUWxU34PAbu2tKbxMN8CmHKg42X2/sAUCubWiiTEAT/ZqORUoFSdnYwEvDTVNs+f/Ht77p/TqNiSaYFuNHzIJ1M6RbBwyOASQfx4P+Sf5E+rtDUYn2fE4AAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAYCAYAAABTPxXiAAAC4UlEQVR4Xu2WWaiNURiGXzOZZZaOIVMiUcgFyXCnEMWNokwlQwrhwpWhKCQiF4QUJUqGSKcoREpJiWRMcSFjIsP79q1lr//b/zn7nNNpn3Nxnnrb+//etf691/9/37cW0ETjpivVzgcbkNZUDx+sjr7UXaqzN0hbqosPlgE90NvUCG/k0Ya6Qy128dnUF+ovddF55WIW9Zzq6Q3Pbuoe1cwbZChsEZu8UUYuU6d8MKUT9Zma5I3AQtgiJnqjjAyjflAjvRFZQz3ywYTDsJRq6Y0yc4Pa54ORa9R5H0x4Anud7amp1LisXWOUqtOp3t4gHZLv46nuyXXkCKp52M9gNZFHH1gqXYctZBVs0bfSQTVAb/EctZX6RY1KvF3Ug/A91t/Zgv2fDTCvmzdaUD+pFd4ILIBN1AI0VswIsTFxUA1YHzQcNndJ4r1DoWhbUa9hv+eZC5tbVBcVwZjmjcAh6huye8ci2JzRSawUO2A9fy1s7oAQj4taFq7FRmpPch3R72nsFG+MDYY+83gMS5+U49QH5LfjUmgvup9cL4f9vtIoshrWET39YWNVVxkGBmOeN2Cbi7zNSUyb4kdYkdWWXrD7bUliSiOlU4qaTN7pYDJs/hBvdAxG3kamhclL9485Iab0q4ClSWQQ7HhSFepsmjsziT2kribX/agzyXWK6ug37DxVxBvqqA+S/dRXWLFFdlLfYTc6gMIGqDf6B9aOq0Jt9BMKTUQPQfvPCxRSUzVY1TlpO/XKByPHqEoXEzepky42gXoLa4FqeRHt+qofLXpwEvco39XrtYFWwt62utFp6gKyXcujN3TJByO6kQrVp4JqormLCXUZ7R95HES2SPPQU9dCY8sWSsWi/p+g//EUls65aICeYlpwdUF/7ooP1hNLqZfILrwIHXeVnyqsurKOWumD9YA61Xtqvjfy0JHghA/WAi2gLntHKfYi2wVLsg35h6+GQicFna3yarOJRsM/oYuEup3/nw0AAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAYCAYAAACiNE5vAAACo0lEQVR4Xu2W24tOURjGH3JKiTASSoQSJVOGC6WkXOACIXEnh8SVSTOk+QMUITmluZAkcrhARCaHkqIQFw4ZEkpRiBun5+ld33zre+29Z++pcTH2r572zPvsvdb7fmuvd22gpKTkf2Iu9YJqp16G61NqfvCvUc+pZ5H2Ba+7OUK9gs2pHJXfyuCdhOVa8XSdFbzp1Ovg65knIZ7Ieeo3Nc8bZDnMa6X6Oq+70QJo7gtUH+dtCt4xaqDzxlAfqWVUP+fV8J76gr8HF7tgEyz1xj9gKmzui96AFSXvkDdgi9Xkg55JsAEueSNwj/pFDfNGCuN8wDGIGuqDKYyC5XbHxXuFmLxTzutPXQ3XTNbCBtjqDTKE+kk98EYGm6m9sOQ8w6k2aoqLpzEAlpv6TsxqWA+Qpz4UozqWuFgix2EDrIKt/kRqQtCa4O3puDsf26mDqC1eRd+gGqJYHr5RH6L/tZ9VrPaxcrsfeSNg/SoXb2CDX450JUh7X4Mv7rg7Pztg+0/Fq+ib1MyaO/Kh/H6g+iO2UAthjVa5tYe4OADr6p2i1dXD6ppJFN3fHhXfSt1C9bgpykNYjoOp0dTZyPtKfQ5/T6MOR14m65C+vzVR0f3tGQk7i88hec/noQ2W43jqKKzTV9B5LU+rrzk0Xy5OwB5MegUXoWv7u4KSUOfV2I2wZtSV4s/A8thA7XeeFkXeemqb8zLJOr93wwbN1SEdcdEVtqC654ugVVYeenPqnHc9eI9gJ0Au1AT0kJqZpzf1GOb7yTpDnfU2NcMbsOLVgIoUvxOWh94az2mYt8IbScyBfX9/gnVzrbh+zQWwgu9Sb4P3nXqH5K+jNPTFlNVZN1KzfTCDZli+SZ+eOjLVOHsk+raY7IOBemqsD5aUlJSU9HT+APGNnB72FsNVAAAAAElFTkSuQmCC>