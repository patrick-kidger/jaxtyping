# Advanced features

## Creating your own dtypes

::: jaxtyping.AbstractDtype
    options:
        members: []

::: jaxtyping.make_numpy_struct_dtype

## Printing axis bindings

::: jaxtyping.print_bindings

## Introspection

If you're writing your own type hint parser, then you may wish to detect if some Python object is a jaxtyping-provided type.

You can check for dtypes by doing `issubclass(x, AbstractDtype)`. For example, `issubclass(Float32, AbstractDtype)` will pass.

You can check for arrays by doing `issubclass(x, AbstractArray)`. Here, `AbstractArray` is the base class for all shape-and-dtype specified arrays, e.g. it's a base class for `Float32[Array, "foo"]`.

You can check for pytrees by doing `issubclass(x, PyTree)`. For example, `issubclass(PyTree[int], PyTree)` will pass.
