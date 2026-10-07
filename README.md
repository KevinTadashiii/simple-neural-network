# Simple Neural Network

A minimal single-neuron neural network built from scratch in Python. The network learns to distinguish between "small" and "large" numbers using gradient descent and the sigmoid activation function.

## Overview

This project implements the simplest possible AI neural network: a single artificial neuron with one weight and one bias. It is trained to classify numbers as either **small** (0) or **large** (1).

### Problem Definition

- Numbers `<= n / 2` are labeled **small** (`0`)
- Numbers `> n / 2` are labeled **large** (`1`)

where `n = 100`.

## How It Works

### Architecture

- **1 input** (normalized number between 0 and 1)
- **1 weight** (`w`) — initialized to `0.0`
- **1 bias** (`b`) — initialized to `0.0`
- **Sigmoid activation** — squashes the weighted input into the range `(0, 1)`

### Training Algorithm

The model is trained using **mean squared error (MSE) loss** and **gradient descent**:

1. **Forward pass** — predict the output for each training sample.
2. **Compute error** — `error = target - output`.
3. **Compute gradients** — derivative of the loss with respect to `weight` and `bias`, using the chain rule through the sigmoid function.
4. **Update parameters** — adjust `weight` and `bias` in the direction that reduces the loss.

This loop runs for `100,000` epochs over the training data.

## Project Structure

```
simple-neural-network/
├── pyproject.toml
├── uv.lock
├── README.md
└── src/
    └── simple_neural_network/
        ├── __init__.py
        └── main.py
```

## Getting Started

### Prerequisites

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) (the project's build tool and dependency manager)

### Installation

```bash
uv sync
```

This installs dependencies and sets up the virtual environment.

### Running the Program

```bash
uv run simple-neural-network
```

Or run the module directly:

```bash
uv run python -m simple_neural_network.main
```

## Output

The program prints the loss every 1,000 epochs during training, then tests the trained neuron on a set of sample inputs:

```
Epoch: 0 Loss: 25.0
Epoch: 1000 Loss: 0.001234
...
Input: 1 Prediction: 0
Input: 45 Prediction: 0
Input: 50 Prediction: 0
Input: 51 Prediction: 1
Input: 65 Prediction: 1
Input: 87 Prediction: 1
```

## Modifying the Model

- **`n`** — controls the range of numbers and the classification threshold (`n / 2`).
- **`learning_rate`** — step size for gradient descent. Larger values converge faster but may overshoot; smaller values are more stable but slower.
- **Number of epochs** — total training iterations. Increase for better convergence.
- **`weight` and `bias`** — initial parameter values.

## License

MIT