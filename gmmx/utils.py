import jax


class register_dataclass_jax:
    """Decorator to register a dataclass with JAX."""

    def __init__(self, data_fields=None, meta_fields=None):
        self.data_fields = data_fields or []
        self.meta_fields = meta_fields or []

    def __call__(self, cls):
        jax.tree_util.register_dataclass(
            cls,
            data_fields=self.data_fields,
            meta_fields=self.meta_fields,
        )
        return cls
