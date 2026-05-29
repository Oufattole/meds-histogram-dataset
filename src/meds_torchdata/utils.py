"""Contains simple utilities used in this package."""

from numpy.random import Generator, default_rng

SEED_OR_RNG = int | Generator | None


def resolve_rng(rng: SEED_OR_RNG) -> Generator:
    """Resolve a random number generator from a seed or generator.

    Args:
        rng: Random number generator for random sampling. If None, a new generator is created. If an
            integer, a new generator is created with that seed.

    Returns:
        A random number generator.

    Raises:
        ValueError: If the random number generator is not a valid type.

    Examples:
        >>> rng = resolve_rng(None)
        >>> isinstance(rng, Generator)
        True

        You can pass a seed, at which point it is deterministic.

        >>> rng = resolve_rng(1)
        >>> isinstance(rng, Generator)
        True
        >>> rng.random()
        0.5118216247002567
        >>> rng.random()
        0.9504636963259353
        >>> resolve_rng(1).random()
        0.5118216247002567
        >>> resolve_rng(2).random()
        0.2616121342493164

        You can also pass a generator directly.

        >>> resolve_rng(default_rng(1)).random()
        0.5118216247002567

        Passing an invalid type raises an error.

        >>> resolve_rng("foo")
        Traceback (most recent call last):
            ...
        ValueError: Invalid random number generator: foo!
    """

    match rng:
        case None:
            return default_rng()
        case int():
            return default_rng(rng)
        case Generator():
            return rng
        case _:
            raise ValueError(f"Invalid random number generator: {rng}!")
