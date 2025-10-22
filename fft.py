import numpy as np
from scipy.fft import fft as scipy_fft

def fft(x:list[complex]) -> np.ndarray[complex]:
    if len(x) <= 1:
        return np.array(x)
    n = len(x)
    # Preencher com zeros para a próxima potência de 2.
    if n & (n - 1) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x = x + [0] * (next_pow2 - n)
        n = next_pow2
    X_even = fft(x[0::2])
    X_odd = fft(x[1::2])
    X = np.zeros(n, dtype=complex)
    for k in range(n):
        Wk = np.exp(-2j * np.pi * k / n)
        X[k] = X_even[k % (n // 2)] + Wk * X_odd[k % (n // 2)]
    return X

def main():
    x = [1, 2, 3, 4, 0, 0, 0, 0]
    fft_result = fft(x)
    print("FFT Result:", fft_result)
    scipy_fft_result = scipy_fft(x)
    print("SciPy FFT Result:", scipy_fft_result)

if __name__ == "__main__":
    main()
