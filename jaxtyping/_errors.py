class TypeCheckError(TypeError):
    pass


# Not inheriting from TypeError as that gets caught and re-reraised as just a TypeError
# when using typeguard<3.
class AnnotationError(Exception):
    pass


TypeCheckError.__module__ = "jaxtyping"
AnnotationError.__module__ = "jaxtyping"
