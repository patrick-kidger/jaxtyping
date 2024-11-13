# Vendored copy of typeguard v2.13.3

We include a vendored copy of typeguard v2.13.3. The reason we need a runtime typechecker is to be able to define `isinstance(..., PyTree[Foo])`.

Of the available options:

- `beartype` does not support `O(n)` checking.
- `typeguard` v4 is notorious for having some bugs (they seem to re-parse the AST or something?? And then they die on the fact that we have strings in our annotations.)
- `typeguard` v2 is what we use here... but we vendor it instead of depending on it, because people may still wish to use typeguard v4 in their own environments. (Notably a number of other packages depend on this, and it's just inconvenient to be incompatible at the package level, when the combinations which don't mix at runtime might never actually be used.)

This is vendored under the terms of the MIT license, which is also reproduced here.
