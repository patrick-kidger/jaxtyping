import subprocess
import sys


_py_path = sys.executable


def test_no_jax_dependency():
    result = subprocess.run(
        f"{_py_path} -c "
        "'import jaxtyping; import sys; sys.exit(\"jax\" in sys.modules)'",
        shell=True,
    )
    assert result.returncode == 0


# Meta-test: test that the above test will work. (i.e. that I haven't messed up using
# subprocess.)
def test_meta():
    result = subprocess.run(
        f"{_py_path} -c 'import jaxtyping; import jax; import sys; "
        'sys.exit("jax" in sys.modules)\'',
        shell=True,
    )
    assert result.returncode == 1
