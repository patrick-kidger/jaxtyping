import pytest
from IPython.testing.globalipapp import start_ipython

from .helpers import ParamError


@pytest.fixture(scope="session")
def session_ip():
    yield start_ipython()


@pytest.fixture(scope="function")
def ip(session_ip):
    session_ip.run_cell(raw_cell="import jaxtyping")
    session_ip.run_line_magic(magic_name="load_ext", line="jaxtyping")
    session_ip.run_line_magic(
        magic_name="jaxtyping.typechecker", line="typeguard.typechecked"
    )
    yield session_ip


def test_that_ipython_works(ip):
    ip.run_cell(raw_cell="x = 1").raise_error()
    assert ip.user_global_ns["x"] == 1


def test_function_beartype(ip):
    ip.run_cell(
        raw_cell="""
    def f(x: int):
        pass
                """
    ).raise_error()
    ip.run_cell(raw_cell="f(1)").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell='f("x")').raise_error()


def test_function_none(ip):
    ip.run_cell(
        raw_cell="""
    def f(a,b,c):
        pass
                """
    ).raise_error()
    ip.run_cell(raw_cell='f(1,2,"k")').raise_error()


def test_function_jaxtyped(ip):
    ip.run_cell(
        raw_cell="""
    from jaxtyping import Float, Array, Int
    import jax

    def g(x: Float[Array, "1"]):
        return x + 1

                """
    ).raise_error()

    ip.run_cell(raw_cell="g(jax.numpy.array([1.0]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="g(jax.numpy.array(1.0))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="g(jax.numpy.array([1]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="g(jax.numpy.array([2, 3]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell='g("string")').raise_error()


def test_function_jaxtyped_and_jitted(ip):
    ip.run_cell(
        raw_cell="""
    from jaxtyping import Float, Array, Int
    import jax

    @jax.jit
    def g(x: Float[Array, "1"]):
        return x + 1

                """
    ).raise_error()

    ip.run_cell(raw_cell="g(jax.numpy.array([1.0]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="g(jax.numpy.array(1.0))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="g(jax.numpy.array([1]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="g(jax.numpy.array([2, 3]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell='g("string")').raise_error()


def test_class_jaxtyped(ip):
    ip.run_cell(
        raw_cell="""
    from jaxtyping import Float, Array, Int
    import equinox as eqx
    import jax

    class A(eqx.Module):
        x: Float[Array, "2"]

        def do_something(self, y: Int[Array, ""]):
            return self.x + y
                """
    ).raise_error()

    ip.run_cell(raw_cell="a = A(jax.numpy.array([1.0, 2.0]))").raise_error()
    ip.run_cell(raw_cell="a.do_something(jax.numpy.array(2))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(raw_cell="A(jax.numpy.array([1.0]))").raise_error()

    with pytest.raises(ParamError):
        ip.run_cell(
            raw_cell="a.do_something(jax.numpy.array([2.0, 3.0]))"
        ).raise_error()


def test_class_not_dataclass(ip):
    ip.run_cell(
        raw_cell="""
    from jaxtyping import Float, Array, Int
    import equinox as eqx
    import jax

    class A:
        def __init__(self, x):
            self.x = x

        def do_something(self, y):
            return x + y
                """
    ).raise_error()

    ip.run_cell(raw_cell="a = A(jax.numpy.array([1.0, 2.0]))").raise_error()
    ip.run_cell(raw_cell="a.do_something(jax.numpy.array(2))").raise_error()
    ip.run_cell(raw_cell="A(jax.numpy.array([1.0]))").raise_error()
    ip.run_cell(raw_cell="a.do_something(jax.numpy.array([2.0, 3.0]))").raise_error()
