def im2col_convolution(batch_of_images, kernels, stride, padding):
    # Pad the batch of images and extract shape of images and shape of kernels
    batch_of_images = np.pad(batch_of_images,((0,0),(0,0),(padding,padding),(padding,padding)))
    batch_size, input_channels, image_height, image_width = batch_of_images.shape
    kernels_number, kernel_channels, kernel_height, kernel_width = kernels.shape

    # Extract sliding windows from the input image, considering the kernel size and stride
    sliding_windows = np.lib.stride_tricks.sliding_window_view(batch_of_images,(1,input_channels,kernel_height,kernel_width))[:,:,::stride,::stride]

    # Reshape the windows to match the unrolled kernel's dimensions
    sliding_windows = sliding_windows.reshape((-1,(kernel_height * kernel_width * input_channels)))

    # Unroll the kernels
    kernels = kernels.reshape((-1,(kernel_height*kernel_width*input_channels))).transpose(1,0)

    # Dot product between images and kernels
    images_dot_kernels = np.matmul(sliding_windows, kernels).astype(np.float32) 

    # Compute the output dimensions to reshape the resulting matrix (each row corresponds to a patch)
    output_width = int(((image_width - kernel_width) / stride) + 1)
    output_height = int(((image_height - kernel_height) / stride) + 1)

    # First operate a reshape keeping spatial ordering, which has channels at the end
    output = images_dot_kernels.reshape(batch_size, output_width, output_height, kernels_number)

    # Transpose to have input in shapes (batch, output_channel, height, width)
    output = output.transpose(0,3,1,2).astype(np.float32)
    return output
