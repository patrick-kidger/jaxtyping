# We have some pretty complicated semantics in `__init__.py`.
# Here we check that we didn't miss one of them on our runtime branch.
def test_all_importable():
    # Ordered according to their appearance in the documentation.
    from jaxtyping import (  # noqa: I001
        Shaped,  # noqa: F401
        Bool,  # noqa: F401
        Key,  # noqa: F401
        Num,  # noqa: F401
        Inexact,  # noqa: F401
        Float,  # noqa: F401
        BFloat16,  # noqa: F401
        Float16,  # noqa: F401
        Float32,  # noqa: F401
        Float64,  # noqa: F401
        Complex,  # noqa: F401
        Complex64,  # noqa: F401
        Complex128,  # noqa: F401
        Integer,  # noqa: F401
        UInt,  # noqa: F401
        UInt2,  # noqa: F401
        UInt4,  # noqa: F401
        UInt8,  # noqa: F401
        UInt16,  # noqa: F401
        UInt32,  # noqa: F401
        UInt64,  # noqa: F401
        Int,  # noqa: F401
        Int2,  # noqa: F401
        Int4,  # noqa: F401
        Int8,  # noqa: F401
        Int16,  # noqa: F401
        Int32,  # noqa: F401
        Int64,  # noqa: F401
        Real,  # noqa: F401
        Array,  # noqa: F401
        ArrayLike,  # noqa: F401
        Scalar,  # noqa: F401
        ScalarLike,  # noqa: F401
        PRNGKeyArray,  # noqa: F401
        PyTreeDef,  # noqa: F401
        PyTree,  # noqa: F401
        jaxtyped,  # noqa: F401
        install_import_hook,  # noqa: F401
        AbstractArray,  # noqa: F401
        AbstractDtype,  # noqa: F401
        print_bindings,  # noqa: F401
        get_array_name_format,  # noqa: F401
        set_array_name_format,  # noqa: F401
    )
