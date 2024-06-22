import numpy as np


class VecValue:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, VecValue) else VecValue(other)
        out = VecValue(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, VecValue) else VecValue(other)
        out = VecValue(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, VecValue) else VecValue(other)
        out = VecValue(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = VecValue(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sum(self):
        out = VecValue(np.sum(self.data), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        other = other if isinstance(other, VecValue) else VecValue(other)
        out = VecValue(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, VecValue) else VecValue(other)
        out = VecValue(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= self.data * out.grad / (other.data * other.data)

        out._backward = _backward

        return out

    def __len__(self):
        return len(self.data)
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class VecLayer:
    def __init__(self, in_features, out_features):
        self.weights = VecValue(np.random.randn(in_features, out_features) * 0.01)
        self.biases = VecValue(np.zeros(out_features))

    def __call__(self, x):
        return x @ self.weights + self.biases

    def parameters(self):
        return [self.weights, self.biases]


class VecMLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = [VecLayer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(pred, target):
    return ((pred - target) * (pred - target)).sum() / VecValue(float(len(target)))


# Example usage
def vec_main(X, Y):

    # Create the model
    model = VecMLP(input_size=2, hidden_sizes=[8, 8], output_size=1)

    # Training loop
    learning_rate = 0.01
    epochs = 1000

    for epoch in range(epochs):
        # Forward pass
        pred = model(VecValue(X))
        loss = mse_loss(pred, VecValue(Y))

        # Backward pass
        loss.backward()

        # Update parameters
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            p.grad = np.zeros_like(p.grad)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")

    # Test the model
    pred = model(VecValue(X))
    print(f"Final predictions: {pred.data}")


# Create a simple dataset
X = np.array([[2.0, 3.0], [-1.0, -2.0], [5.0, -1.0], [-3.0, 4.0]])
Y = np.array([[1.0], [-1.0], [1.0], [-1.0]])
vec_main(X, Y)
