# GAN MNIST Image Generation

A TensorFlow/Keras implementation of Generative Adversarial Networks (GANs) for generating handwritten digit images using the MNIST dataset. This project demonstrates the classic adversarial training approach where a generator and discriminator network compete against each other to create realistic synthetic images.

## Overview

This project implements a Deep Convolutional GAN (DCGAN) that learns to generate new handwritten digits by training two neural networks in an adversarial manner:

- **Generator**: Creates fake images from random noise
- **Discriminator**: Distinguishes between real and generated images

The training process follows the minimax game theory where the generator tries to fool the discriminator, while the discriminator tries to correctly identify real vs. fake images.

## Architecture

### Generator Network
- **Input**: 100-dimensional noise vector
- **Output**: 28×28×1 grayscale images
- **Architecture**:
  - Dense layer (100 → 7×7×256)
  - BatchNormalization + LeakyReLU (α=0.02)
  - Reshape to (7, 7, 256)
  - Conv2DTranspose layers for upsampling (7×7 → 14×14 → 28×28)
  - Final activation: tanh
- **Parameters**: 2,330,944 (8.89 MB)

### Discriminator Network
- **Input**: 28×28×1 grayscale images
- **Output**: Single probability value (real vs. fake)
- **Architecture**:
  - Conv2D layers with 2×2 strides for downsampling
  - LeakyReLU activation (α=0.2)
  - BatchNormalization and Dropout (0.3-0.4)
  - Final Dense layer with sigmoid activation
- **Parameters**: 375,553 (1.43 MB)

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| **Dataset** | MNIST (60,000 training images) |
| **Image Size** | 28×28×1 (grayscale) |
| **Batch Size** | 256 |
| **Epochs** | 100 |
| **Noise Dimension** | 100 |
| **Learning Rate** | 1e-4 (both networks) |
| **Optimizer** | Adam |
| **Loss Function** | Binary Cross-Entropy |

## Features

- 🎯 **Real-time Training Visualization**: Generates sample images during training
- 💾 **Model Checkpointing**: Saves model weights every 15 epochs
- 📊 **Progress Monitoring**: Displays training time per epoch
- 🔄 **Alternating Training**: Balanced training between generator and discriminator
- 🎨 **Image Generation**: Creates 4×4 grid of generated samples

## Requirements

```
tensorflow>=2.0.0
numpy
matplotlib
Pillow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abhaykum123/GAN.git
cd gan-mnist-generator
```

2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib pillow
```

3. Run the notebook:
```bash
jupyter notebook GAN.ipynb
```

## Usage

### Training the Model

The main training loop alternates between:
1. Training the discriminator on real and generated images
2. Training the generator to fool the discriminator

```python
# Key training parameters
EPOCHS = 100
BATCH_SIZE = 256
noise_dim = 100

# Train the model
train(train_dataset, EPOCHS)
```

### Generating Images

After training, you can generate new images using the trained generator:

```python
# Generate random noise
noise = tf.random.normal([16, noise_dim])

# Generate images
generated_images = generator(noise, training=False)
```

## Model Checkpoints

The model automatically saves checkpoints every 15 epochs in the `/content/model/` directory. You can restore training from these checkpoints:

```python
# Restore from checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

## Training Process

The training follows the classic GAN objective function:

```
min_G max_D V(D,G) = E_x~pdata[log D(x)] + E_z~pz[log(1-D(G(z)))]
```

Where:
- G: Generator network
- D: Discriminator network  
- x: Real data samples
- z: Random noise vectors

## Results

The model generates progressively better handwritten digits over the course of training, with sample images saved at each epoch showing the generator's improvement in creating realistic MNIST-style digits.



https://github.com/user-attachments/assets/f5e1e4b8-09cc-4421-b8ec-38194b61cbcd



## Key Implementation Details

- **Gradient Tape**: Uses TensorFlow's automatic differentiation for backpropagation
- **Batch Normalization**: Applied to stabilize training
- **LeakyReLU**: Prevents dying ReLU problem in adversarial training
- **Dropout**: Added to discriminator to prevent overfitting
- **Alternative Loss**: Uses `log D(G(z))` instead of `log(1-D(G(z)))` for better generator gradients early in training

## References

This implementation is based on the original GAN paper:
- Goodfellow, I., et al. "Generative Adversarial Nets." NIPS 2014

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Note**: This implementation is designed for educational purposes and demonstrates the fundamental concepts of GANs using the MNIST dataset. For production use, consider additional techniques like spectral normalization, progressive growing, or more advanced GAN variants.
