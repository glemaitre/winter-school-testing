import pytest

from compute import divide
from compute import multiply


@pytest.mark.parametrize(
    'a, b, x',
    [(1, 2, 0.5),
     (4, 2, 2)]
)
def test_divide(a, b, x):
    res = divide(a, b)
    assert res == pytest.approx(x)


def test_multiply():
    assert multiply(2, 2) == 4
