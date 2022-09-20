from jaxtyping import jaxtyped


class M:
    @jaxtyped
    @classmethod
    def f(cls):
        return 3


# Check that the @jaxtyped decorator doesn't blat the __get__ of @classmethod
def test_decorator():
    assert M.f() == 3
