# FAQ

## `flake8` is throwing an error.

In type annotations, strings are used for two different things. Sometimes they're strings. Sometimes they're "forward references", used to refer to a type that will be defined later.

Some tooling in the Python ecosystem assumes that only the latter is true, and will throw spurious errors if you try to use a string just as a string (like we do).

In the case of `flake8`, at least, this is easily resolved. Multi-dimensional arrays (e.g. `f32["b c"]`) will throw a very unusual error (F722, syntax error in forward annotation), so you can safely just disable this particular error globally. Uni-dimensional arrays (e.g. `f32["x"]`) will throw an error that's actually useful (F821, undefined name), so instead of disabling this globally, you should instead prepend a space to the start of your shape, e.g. `f32[" x"]`. `jaxtyping` will treat this in the same way, whilst `flake8` will now throw an F722 error that you can disable as before.

## What about support for static type checkers, like `mypy`, `pyright`, etc.?

Nope.

Python's static typing ecosystem is a complicated collection of edge cases. Many of them block ML/scientific computing in particular. A few examples:

1. The static type system is intrinsically not expressive enough to describe operations like concatenation, stacking, or broadcasting.

2. Axes have to be lifted to type-level variables. Meanwhile the approach taken in libraries like `jaxtyping` and [TorchTyping](https://github.com/patrick-kidger/torchtyping) is to use value-level variables for types: because that's what the underlying JAX, PyTorch etc. libraries use! As such, making a static type checker work with these libraries would require either fundamentally rewriting these libraries, or exhaustively maintaining type stubs for them, and would *still* require a `typing.cast` any time you use anything unstubbed (e.g. any third party library, or part of your codebase you haven't typed yet). This is a huge maintenance burden for anyone.

3. Static type checkers have a variety of bugs that affect this use case. `mypy` doesn't support `Protocol`s correctly. `pyright` doesn't support genericised subprotocols. etc.

4. Variadic generics exist. Variadic protocols do not. (It's not clear that these have been contemplated.)

5. The syntax for static typing is verbose. You have to write things like `Array[Unpack[AnyShape], Literal[3], Height, Width]` instead of `Array["... 3 height width"]`.

6. [The underlying type system has flaws](https://github.com/patrick-kidger/torchtyping/issues/37#issuecomment-1153294196). [The numeric tower is broken](https://stackoverflow.com/a/69383462); [int is not a number](https://github.com/python/mypy/issues/3186#issuecomment-885718629); [virtual base classes don't work](https://github.com/python/mypy/issues/2922); [complex lies about having comparison operations, so type checkers have to lie about that lie in order to remove them again](https://posita.github.io/numerary/0.4/whytho/); `typing.*` don't work with `isinstance`; co/contra-variance are baked into containers (not specified at use-time); `dict` is variadic despite... not being variadic; bool is a subclass of int (!); ... etc. etc.

## What about [PEP 646](https://www.python.org/dev/peps/pep-0646/) and variadic generics?

[Doesn't change the previous issues, unfortunately.](https://github.com/patrick-kidger/torchtyping/issues/37) All the problems of the previous heading still hold true. They're just also true for types like `AnyDimensionalArray[Batch, Channels, AsManyArgumentsAsWePlease]` as well as types like `TwoDimensionalArray[Batch, Channels]`.

## Is the lack of interaction with static typing a problem?

At least for any software that is mostly just running JAX code, no!

The correct way to use JAX is to put together all your operations, and then put a single `jax.jit` right at the very top. This gives you optimal speed; anything else will be unnecessarily (and substantially) slower.

This means that all the type checking only gets resolved once: at trace time. Afterwards JAX still lowers everything down to the same optimised code.

In some sense, `python myprogram.py` just ends up doing the same as `mypy myprogram.py`. Except instead of throwing away all the work used to parse your code, build the abstract syntax tree, etc. (and requiring you to then run `python myprogram.py` afterwards to actually use it), it can keep it around and just run your code immediately.

TL;DR: `jax.jit` is amazing.
