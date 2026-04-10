import stratoquant as sq
import numpy as np

def test_version():
    assert sq.__version__ == "1.0.0"

def test_bs_price():
    price = sq.black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')
    assert abs(price - 10.4506) < 0.001

def test_delta_vectorized():
    K_grid = np.linspace(80, 120, 41)
    deltas = sq.delta(100, K_grid, 1.0, 0.05, 0.2, 'call')
    assert deltas.shape == (41,)
    assert all(0 <= d <= 1 for d in deltas)
