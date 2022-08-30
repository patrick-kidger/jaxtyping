# FAQ

## `flake8` is throwing an error.

In type annotations, strings are used for two different things. Sometimes they're strings. Sometimes they're "forward references", used to refer to a type that will be defined later.

Some tooling in the Python ecosystem assumes that only the latter is true, and will throw spurious errors if you try to use a string just as a string (like we do).

In the case of `flake8`, at least, this is easily resolved. Multi-dimensional arrays (e.g. `f32[Array, "b c"]`) will throw a very unusual error (F722, syntax error in forward annotation), so you can safely just disable this particular error globally. Uni-dimensional arrays (e.g. `f32[Array, "x"]`) will throw an error that's actually useful (F821, undefined name), so instead of disabling this globally, you should instead prepend a space to the start of your shape, e.g. `f32[Array, " x"]`. `jaxtyping` will treat this in the same way, whilst `flake8` will now throw an F722 error that you can disable as before.

## Does jaxtyping work with static type checkers like `mypy`/`pyright`/`pytype`?

There is partial support for these. An annotation of the form `dtype[array, shape]` should be treated as just `array` by a static type checker. Unfortunately full dtype/shape checking is beyond the scope of what static type checking is currently capable of.

(Note that at time of writing, `pytype` has a bug in that `dtype[array, shape]` is sometimes treated as `Any` rather than `array`. The other two work fine.)

## How does jaxtyping interact with `jax.jit`?

jaxtyping and `jax.jit` synergise beautifully.

When calling JAX operations wrapped in a `jax.jit`, then the dtype/shape-checking will happen at trace time. (When JAX traces your function prior to compiling it.) The actual compiled code does not have any dtype/shape-checking, and will therefore still be just as fast as before!

## Does jaxtyping use [PEP 646](https://www.python.org/dev/peps/pep-0646/) (variadic generics)?

The intention of PEP 646 was to make it possible for static type checkers to perform shape checks of arrays. Unfortunately, this still isn't yet practical, so jaxtyping deliberately does not use this. (Yet?)

The real problem is that Python's static typing ecosystem is a complicated collection of edge cases. Many of them block ML/scientific computing in particular. For example:

1. The static type system is intrinsically not expressive enough to describe operations like concatenation, stacking, or broadcasting.

2. Axes have to be lifted to type-level variables. Meanwhile the approach taken in libraries like `jaxtyping` and [TorchTyping](https://github.com/patrick-kidger/torchtyping) is to use value-level variables for types: because that's what the underlying JAX, PyTorch etc. libraries use! As such, making a static type checker work with these libraries would require either fundamentally rewriting these libraries, or exhaustively maintaining type stubs for them, and would *still* require a `typing.cast` any time you use anything unstubbed (e.g. any third party library, or part of your codebase you haven't typed yet). This is a huge maintenance burden.

3. Static type checkers have a variety of bugs that affect this use case. `mypy` doesn't support `Protocol`s correctly. `pyright` doesn't support genericised subprotocols. etc.

4. Variadic generics exist. Variadic protocols do not. (It's not clear that these were contemplated.)

5. The syntax for static typing is verbose. You have to write things like `Array[Float32, Unpack[AnyShape], Literal[3], Height, Width]` instead of `f32[Array, "... 3 height width"]`.

6. [The underlying type system has flaws](https://github.com/patrick-kidger/torchtyping/issues/37#issuecomment-1153294196). [The numeric tower is broken](https://stackoverflow.com/a/69383462); [int is not a number](https://github.com/python/mypy/issues/3186#issuecomment-885718629); [virtual base classes don't work](https://github.com/python/mypy/issues/2922); [complex lies about having comparison operations, so type checkers have to lie about that lie in order to remove them again](https://posita.github.io/numerary/0.4/whytho/); `typing.*` don't work with `isinstance`; co/contra-variance are baked into containers (not specified at use-time); `dict` is variadic despite... not being variadic; bool is a subclass of int (!); ... etc. etc.
