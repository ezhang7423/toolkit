import unittest
import math
from main import Value, Plus, Mult, Sub, Pow, ReLU, Neg, Divide


class TestAutogradEngine(unittest.TestCase):

    def setUp(self):
        # Reset Value._id before each test
        Value._id = 0

    def test_basic_operations(self):
        # Test addition
        a = Value(2)
        b = Value(3)
        c = a + b
        c.backward()
        self.assertEqual(c.value, 5)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

        # Test multiplication
        a = Value(2)
        b = Value(3)
        c = a * b
        c.backward()
        self.assertEqual(c.value, 6)
        self.assertEqual(a.grad, 3)
        self.assertEqual(b.grad, 2)

        # Test subtraction
        a = Value(5)
        b = Value(3)
        c = a - b
        c.backward()
        self.assertEqual(c.value, 2)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, -1)

        # Test division
        a = Value(6)
        b = Value(2)
        c = a / b
        c.backward()
        self.assertEqual(c.value, 3)
        self.assertEqual(a.grad, 0.5)
        self.assertEqual(b.grad, -1.5)

    def test_power_operation(self):
        a = Value(2)
        b = a**3
        b.backward()
        self.assertEqual(b.value, 8)
        self.assertEqual(a.grad, 12)

    def test_relu_operation(self):
        # Test ReLU with positive input
        a = Value(2)
        b = a.relu()
        b.backward()
        self.assertEqual(b.value, 2)
        self.assertEqual(a.grad, 1)

        # Test ReLU with negative input
        a = Value(-2)
        b = a.relu()
        b.backward()
        self.assertEqual(b.value, 0)
        self.assertEqual(a.grad, 0)

    def test_neg_operation(self):
        a = Value(2)
        b = -a
        b.backward()
        self.assertEqual(b.value, -2)
        self.assertEqual(a.grad, -1)

    def test_complex_expression(self):
        x = Value(2)
        y = Value(3)
        z = Value(4)
        expr = (x * y + z.relu()) / (x - y) ** 2
        expr.backward()
        self.assertAlmostEqual(expr.value, 10)
        self.assertAlmostEqual(x.grad, 23)
        self.assertAlmostEqual(y.grad, -18)
        self.assertAlmostEqual(z.grad, 1)

    def test_repeated_operations(self):
        x = Value(2)
        y = x + x + x
        y.backward()
        self.assertEqual(y.value, 6)
        self.assertEqual(x.grad, 3)

    def test_branching(self):
        x = Value(2)
        a = x * Value(3)
        b = x + Value(1)
        y = a * b
        y.backward()
        self.assertEqual(y.value, 18)
        self.assertEqual(x.grad, 15)

    def test_zero_gradient(self):
        x = Value(2)
        y = Value(3)
        z = x * y
        w = z - z
        w.backward()
        self.assertEqual(w.value, 0)
        self.assertEqual(x.grad, 0)
        self.assertEqual(y.grad, 0)

    def test_division_by_zero(self):
        x = Value(1)
        y = Value(0)
        with self.assertRaises(ZeroDivisionError):
            z = x / y

    def test_gradients_accumulation(self):
        x = Value(2)
        y = x * x
        y.backward()
        self.assertEqual(x.grad, 4)
        y.backward()
        self.assertEqual(x.grad, 8)  # Gradients should accumulate
        
    # TODO        
    # def test_higher_order_gradients(self):
    #     x = Value(2)
    #     y = x * x * x
    #     y.backward()
    #     self.assertEqual(x.grad, 12)

    #     # Reset gradients
    #     x.grad = None
    #     y.grad = None

    #     # Compute second-order gradient
    #     y.backward()
    #     x_grad = x.grad
    #     x_grad.backward()
    #     self.assertEqual(x.grad, 36)  # d²y/dx² = 6x = 6 * 2 = 12


if __name__ == "__main__":
    unittest.main()
