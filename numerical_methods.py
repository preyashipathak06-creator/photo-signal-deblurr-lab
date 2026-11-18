
import numpy as np


def build_blur_matrix(size):
    B = np.zeros((size, size))

    for i in range(size):
        for j in (i - 1, i, i + 1):
            B[i, min(max(j, 0), size - 1)] += 1 / 3

    return B


def gauss_elimination(A, b):
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    vector = b.ndim == 1
    if vector:
        b = b.reshape(-1, 1)

    n = len(A)

    for i in range(n):
        pivot = i + np.argmax(np.abs(A[i:, i]))

        if abs(A[pivot, i]) < 1e-12:
            raise ValueError("Matrix is singular.")

        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros_like(b)

    for i in range(n - 1, -1, -1):
        x[i] = (
            b[i] - A[i, i + 1:].T @ x[i + 1:]
        ) / A[i, i]

    return x[:, 0] if vector else x


def gauss_jordan(A, b):
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    vector = b.ndim == 1
    if vector:
        b = b.reshape(-1, 1)

    n = len(A)

    for i in range(n):
        pivot = i + np.argmax(np.abs(A[i:, i]))

        if abs(A[pivot, i]) < 1e-12:
            raise ValueError("Matrix is singular.")

        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]

        pivot_value = A[i, i]

        A[i] /= pivot_value
        b[i] /= pivot_value

        for j in range(n):
            if i != j:
                factor = A[j, i]
                A[j] -= factor * A[i]
                b[j] -= factor * b[i]

    return b[:, 0] if vector else b


def direct_restore(B, image, method="gauss"):
    solve = gauss_elimination if method == "gauss" else gauss_jordan

    temp = solve(B, image)
    restored = solve(B, temp.T).T

    return restored


def regularized_least_squares(B, image, lam=0.025):
    n = B.shape[0]

    A = B.T @ B + lam * np.eye(n)
    right = B.T @ image

    temp = gauss_elimination(A, right)
    restored = gauss_elimination(A, B.T @ temp.T).T

    return restored


def power_method(A, iterations=100):
    rng = np.random.default_rng(123)
    x = rng.normal(size=A.shape[0])
    x /= np.linalg.norm(x)

    eigenvalue = 0

    for _ in range(iterations):
        y = A @ x
        x = y / np.linalg.norm(y)

        new_value = float(x @ A @ x)

        if abs(new_value - eigenvalue) < 1e-10:
            eigenvalue = new_value
            break

        eigenvalue = new_value

    return abs(eigenvalue)


def inverse_power_method(A, iterations=100):
    rng = np.random.default_rng(456)
    x = rng.normal(size=A.shape[0])
    x /= np.linalg.norm(x)

    eigenvalue = 0

    for _ in range(iterations):
        y = gauss_elimination(A, x)
        x = y / np.linalg.norm(y)

        new_value = float(x @ A @ x)

        if abs(new_value - eigenvalue) < 1e-10:
            eigenvalue = new_value
            break

        eigenvalue = new_value

    return abs(eigenvalue)


def estimate_condition_number(A):
    largest = power_method(A)
    smallest = inverse_power_method(A)

    condition = largest / smallest if smallest > 1e-14 else float("inf")

    return largest, smallest, condition
