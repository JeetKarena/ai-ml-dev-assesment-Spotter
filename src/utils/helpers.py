"""General-purpose helper functions."""


def to_snake_case(name: str) -> str:
    """Convert a CamelCase or PascalCase identifier to snake_case."""
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
