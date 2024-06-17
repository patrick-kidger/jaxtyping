import subprocess
import sys
import unittest


_py_path = sys.executable


@unittest.skipIf(not _py_path, "test requires sys.executable")
def test_no_jax_dependency():
    result = subprocess.run(
        f"{_py_path} -c "
        "'import jaxtyping; import sys; sys.exit(\"jax\" in sys.modules)'",
        shell=True,
    )
    assert result.returncode == 0


# Meta-test: test that the above test will work. (i.e. that I haven't messed up using
# subprocess.)
@unittest.skipIf(not _py_path, "test requires sys.executable")
def test_meta():
    result = subprocess.run(
        f"{_py_path} -c 'import jaxtyping; import jax; import sys; "
        'sys.exit("jax" in sys.modules)\'',
        shell=True,
    )
    assert result.returncode == 1
