# CNN Convolution: Im2Col and Optimization Approaches

This repository explores and compares various methods for CNN convolution operations, primarily focusing on the **Im2Col (image-to-column)** technique. Implementations are in NumPy and CuPy, benchmarked against a PyTorch CNN on the MNIST dataset.

## Core Focus

*   **Im2Col Technique:** Transforming 2D convolution into efficient matrix multiplication.
*   **Implementation & Comparison:**
    *   Naive nested-loop NumPy.
    *   Im2Col in NumPy (basic and optimized, including BLAS).
    *   Im2Col in CuPy (GPU-accelerated).
    *   PyTorch CNN (reference).
*   **Key Concepts:** Padding, matrix dilation, forward/backward passes for conv & MLP layers.
*   **Benchmarking:** Inference and training performance analysis.

## How to Run

1.  **Clone the Repository.**
2.  **Install Dependencies using** `requirements.txt`:

    All code was executed using a PC with a NVIDIA RTX 3050 wiith CUDA 12.8. Please check which version of CUDA your device has installed and edit accordingly the Pytorch and CuPy versions in the requirements.
4.  **Dataset & Weights:**
    *   Place MNIST dataset files in a `MNIST/` directory.
    *   Ensure `simple_cnn_mnist.pth` (pre-trained PyTorch weights) is in the root.
    *   The `images/` directory is needed for markdown visuals.
5.  **Launch Jupyter the way you prefer**
    Execute cells sequentially. CuPy cells require a compatible NVIDIA GPU.
