from graphviz import Digraph

#################
# Operations
#################


class Plus:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.z = None

    def forward(self):
        self.z = Value(self.x.value + self.y.value, self)
        return self.z

    def backward(self):

        if self.x.grad is None:
            self.x.grad = 0

        if self.y.grad is None:
            self.y.grad = 0

        self.x.grad += 1 * self.z.grad
        self.y.grad += 1 * self.z.grad

        self.x.backward()
        self.y.backward()

    def __repr__(self) -> str:
        return f"Plus({self.x}, {self.y})"


class Mult:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.z = None

    def forward(self):
        self.z = Value(self.x.value * self.y.value, self)
        return self.z

    def backward(self):

        if self.x.grad is None:
            self.x.grad = 0

        if self.y.grad is None:
            self.y.grad = 0

        self.x.grad += self.y.value * self.z.grad
        self.y.grad += self.x.value * self.z.grad

        self.x.backward()
        self.y.backward()

    def __repr__(self) -> str:
        return f"Mult({self.x}, {self.y})"


class ReLU:
    def __init__(self, x) -> None:
        self.x = x
        self.z = None

    def forward(self):
        self.z = Value(max(0, self.x.value), self)
        return self.z

    def backward(self):
        if self.x.grad is None:
            self.x.grad = 0

        self.x.grad += (1 if self.x.value > 0 else 0) * self.z.grad
        self.x.backward()

    def __repr__(self) -> str:
        return f"ReLU({self.x})"


class Neg:
    def __init__(self, x) -> None:
        self.x = x
        self.z = None

    def forward(self):
        self.z = Value(-self.x.value, self)
        return self.z

    def backward(self):
        if self.x.grad is None:
            self.x.grad = 0

        self.x.grad += -1 * self.z.grad
        self.x.backward()

    def __repr__(self) -> str:
        return f"Neg({self.x})"


class Divide:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.z = None

    def forward(self):
        self.z = Value(self.x.value / self.y.value, self)
        return self.z

    def backward(self):
        if self.x.grad is None:
            self.x.grad = 0

        if self.y.grad is None:
            self.y.grad = 0

        self.x.grad += (1 / self.y.value) * self.z.grad
        self.y.grad += (-self.x.value / (self.y.value**2)) * self.z.grad

        self.x.backward()
        self.y.backward()

    def __repr__(self) -> str:
        return f"Divide({self.x}, {self.y})"


class Sub:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.z = None

    def forward(self):
        self.z = Value(self.x.value - self.y.value, self)
        return self.z

    def backward(self):

        if self.x.grad is None:
            self.x.grad = 0

        if self.y.grad is None:
            self.y.grad = 0

        self.x.grad += 1 * self.z.grad
        self.y.grad += -1 * self.z.grad

        self.x.backward()
        self.y.backward()

    def __repr__(self) -> str:
        return f"Sub({self.x}, {self.y})"


#################
# Value
#################


class Value:
    _id = 0

    def __init__(self, value, op=None) -> None:
        self.id = Value._id
        Value._id += 1
        self.grad = None
        self.value = value
        self.op = op

    def backward(self):
        if self.grad is None:
            self.grad = 1

        if self.op is None:
            return
        self.op.backward()

    def __add__(self, other):
        return Plus(self, other).forward()

    def __mul__(self, other):
        return Mult(self, other).forward()

    def __repr__(self) -> str:
        return f"Value({self.value}, grad={self.grad}, op={self.op})"

    def __pow__(self, other):
        return Pow(self, other).forward()

    def relu(self):
        return ReLU(self).forward()

    def __neg__(self):
        return Neg(self).forward()

    def __truediv__(self, other):
        return Divide(self, other).forward()

    def __rtruediv__(self, other):
        return Divide(Value(other), self).forward()
    
    def __sub__(self, other):
        return Sub(self, other).forward()


class Pow:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.z = None

    def forward(self):
        self.z = Value(self.x.value**self.y, self)
        return self.z

    def backward(self):
        if self.x.grad is None:
            self.x.grad = 0

        self.x.grad += (self.y * self.x.value ** (self.y - 1)) * self.z.grad
        self.x.backward()

    def __repr__(self) -> str:
        return f"Pow({self.x}, {self.y})"


#################
# Utils
#################


# Update the visualize function to include new operations
def visualize(root):
    dot = Digraph(comment="Computation Graph")
    dot.attr(rankdir="TB")

    def add_nodes(node):
        if isinstance(node, Value):
            dot.node(str(node.id), f"val:{node.value}\ngrad: {node.grad}")
            if node.op:
                op_id = f"op_{node.id}"
                op_symbol = {
                    Plus: "+",
                    Mult: "*",
                    Sub: "-",
                    Pow: "^",
                    ReLU: "ReLU",
                    Neg: "-",
                    Divide: "/",
                }.get(type(node.op), "Unknown")

                dot.node(op_id, op_symbol)
                dot.edge(op_id, str(node.id))

                if hasattr(node.op, "x"):
                    dot.edge(str(node.op.x.id), op_id)
                    add_nodes(node.op.x)

                if hasattr(node.op, "y"):
                    try:
                        dot.edge(str(node.op.y.id), op_id)
                    except AttributeError:
                        dot.edge(str(node.op.y), op_id)
                    add_nodes(node.op.y)

    add_nodes(root)
    return dot
