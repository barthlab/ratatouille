import functools
from typing import Any, Callable


class value_dispatch:
    """
    A decorator that transforms a function into a generic function dispatching
    on the value of its first argument.
    """
    def __init__(self, func: Callable):
        self.registry: dict[Any, Callable] = {}
        self.default_func: Callable = func        
        functools.update_wrapper(self, func)

    def register(self, value):
        """Decorator to register a function for a specific value."""
        def decorator(func):
            self.registry[value] = func
            return func
        return decorator

    def __call__(self, first_arg, *args, **kwargs):
        """Call the appropriate registered function or the default."""
        func = self.registry.get(first_arg, self.default_func)
        return func(first_arg, *args, **kwargs)