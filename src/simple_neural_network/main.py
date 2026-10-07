# The simplest AI neural network
# Objective: to learn to distinguish between small and large numbers.

# Training Data:
# Number <= n / 2 is considered "small" (0)
# Number > n / 2 is considered "large" (1)

import math

n = 100
data = [(i / 100, 0 if i <= n // 2 else 1) for i in range(1, n + 1)]

# Neuron parameters
weight = 0.0
bias = 0.0

learning_rate = 0.01

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Function to make a prediction
def predict(x):
    return sigmoid(x * weight + bias)


def train():
    global weight, bias

    # Training
    for epoch in range(10000):

        total_loss = 0

        for x, target in data:

            # 1. The neuron makes a prediction
            output = predict(x)

            # 2. Calculate the error
            error = target - output
            loss = error ** 2

            total_loss += loss

            gradient_weight = (
                2
                * error
                * output
                * (1 - output)
                * x
            )

            gradient_bias = (
                2
                * error
                * output
                * (1 - output)
            )

            # 3. Adjust weight
            weight += learning_rate * gradient_weight

            # 4. Adjust bias
            bias += learning_rate * gradient_bias

        if epoch % 1000 == 0:
            print(
                "Epoch:", epoch,
                "Loss:", total_loss
            )


def test():
    # Do the test
    for x in [1, 45, 50, 51, 65, 87]:
        normalized_x = x / n

        prediction = predict(normalized_x)

        if prediction < 0.5:
            prediction = 0
        else:
            prediction = 1

        print(
            "Input:", x,
            "Prediction:", prediction
        )


def main():
    train()
    test()


if __name__ == "__main__":
    main()