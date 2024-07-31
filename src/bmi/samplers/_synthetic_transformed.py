from typing import Callable, Optional, TypeVar, Union, List

import jax
import jax.numpy as jnp
import numpy as np

import bmi.samplers.base as base
from bmi.interface import ISampler, KeyArray



SomeArray = Union[jnp.ndarray, np.ndarray]
Transform = Callable[[SomeArray], jnp.ndarray]

_T = TypeVar("_T")


def identity(x: _T) -> _T:
    """The identity mapping."""
    return x
    

class TransformedSampler(base.BaseSampler):
    """
    Pushforward of a distribution $P_{XY}$
    via a product mapping
        $f \\times g$.

    In other words, we have mutual information between $f(X)$ and $g(Y)$
    for some mappings $f$ and $g$.

    Note:
        By default we assume that f and g are diffeomorphisms, so that
            I(f(X); g(Y)) = I(X; Y).
        If you don't use diffeomorphisms (in particular,
        non-default `add_dim_x` or `add_dim_y`), overwrite the
        `mutual_information()` method
    """

    def __init__(
        self,
        base_sampler: ISampler,
        transform_x: Optional[Callable] = None,
        transform_y: Optional[Callable] = None,
        add_dim_x: int = 0,
        add_dim_y: int = 0,
        vectorise: bool = True,
        ) -> None:
        """
        Parameters
        ----------
        base_sampler: 
            allows sampling from $P(X, Y)$
        transform_x: 
            diffeomorphism $f$, so that we have variable $f(X)$.
            By default the identity mapping.
        transform_y: 
            diffeomorphism $g$, so that we have variable $g(Y)$.
            By default the identity mapping.
        add_dim_x: 
            the difference in dimensions of the output of $f$ and its input. 
            Note that for any diffeomorphism it must be zero
        add_dim_y: 
            similarly as `add_dim_x`, but for $g$.
        vectorise: 
            whether to use `jax.vmap` to vectorise transforms.
            If False, provided `transform_X` and `transform_Y` need to already be vectorized.

        Note:
            If you don't use diffeomorphisms (in particular,
            non-default `add_dim_x` or `add_dim_y`), overwrite the
            `mutual_information()` method
        """
        # Check if the dimensions added are valid (non-negative)
        if add_dim_x < 0 or add_dim_y < 0:
            raise ValueError("Transformed samplers cannot decrease dimensionality.")

        super().__init__(dim_x=base_sampler.dim_x + add_dim_x, dim_y=base_sampler.dim_y + add_dim_y)
        
        # Set default transformations to identity if not provided
        if transform_x is None:
            transform_x = identity
        if transform_y is None:
            transform_y = identity
        
        # We either vectorise the transform or assume that we were given vectorised transforms
        self._vectorized_transform_x = (jax.vmap(transform_x) if vectorise else transform_x)
        self._vectorized_transform_y = (jax.vmap(transform_y) if vectorise else transform_y)
        self._base_sampler = base_sampler


    def transform(
        self,
        x: SomeArray,
        y: SomeArray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Transforms given samples by `f x g`.

        Parameters
        ----------
        x: samples, (n_points, dim(X))
        y: samples, (n_points, dim(Y))

        Returns
        -------
        f(x), shape (n_points, dim(X) + add_dim_x)
        g(y), shape (n_points, dim(Y) + add_dim_y)
        """
        return self._vectorized_transform_x(x), self._vectorized_transform_y(y)
    
    def sample(
        self,
        n_points: int,
        rng: Union[int, KeyArray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Samples from the distribution $P(f(X), g(Y))$.
        
        Parameters
        ----------
        n_points : int
            The number of points to sample.
        rng : Union[int, KeyArray]
            Random number generator seed or key.

        Returns
        -------
        paired samples
            from $f(X)$, shape `(n_points, dim(X) + add_dim_x)` and
            from $g(Y)$, shape `(n_points, dim(Y) + add_dim_y)`
        """
        x, y = self._base_sampler.sample(n_points=n_points, rng=rng)
        return self.transform(x, y)
    
    def sample_split(
        self,
        n_points: int,
        rng: Union[int, KeyArray],
        transformer: str,
        spiral_dims: Optional[List[int]] = None
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Samples from the distribution $P(f(X), g(Y))$ and applies the specified transformer.

        Parameters
        ----------
        n_points : int
            The number of points to sample.
        rng : Union[int, KeyArray]
            Random number generator seed or key.
        transformer : str
            The type of transformer to apply. Options are 'wiggly' and 'spiral'.
        spiral_dims : Optional[List[int]]
            A list of dimensions for the spiral transformation. Required if transformer is 'spiral'.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Paired samples
            from $f(X)$, shape `(n_points, dim(X) + add_dim_x)` and
            from $g(Y)$, shape `(n_points, dim(Y) + add_dim_y)`.

        Raises
        ------
        ValueError: If transformer is 'wiggly' and input x is not 2-dimensional.
        ValueError: If transformer is 'spiral' and spiral_dims is not provided.
        ValueError: If transformer is 'spiral' and input x has fewer than 5 dimensions.
        """
        # Sample from the base distribution
        x, y = self._base_sampler.sample(n_points=n_points, rng=rng)
        
        if transformer.lower() == "wiggly":
            if x.shape[1] != 2:
                raise ValueError("Input x must be 2-dimensional for wiggly transformer.")
            
            # Transform x by iteratively applying the transformation function
            x_transformed1, x_transformed2 = self.transform(x[:, 0], x[:, 1])
            x_transformed1 = x_transformed1.reshape(-1, 1)
            x_transformed2 = x_transformed2.reshape(-1, 1)
            x_transformed = np.concatenate((x_transformed1, x_transformed2), axis=1)
            return x_transformed, y
        
        elif transformer.lower() == "spiral":
            # Check if spiral_dims is provided and x has at least 5 dimensions
            if spiral_dims is None:
                raise ValueError("spiral_dims must be provided for spiral diffeomorphism transformer.")
            if x.shape[1] < 5:
                raise ValueError("Input x must have at least 5 dimensions for spiral diffeomorphism transformer.")
            
            # Apply spiral transformation based on specified dimensions
            x_transformed1 = x[:, :spiral_dims[0]]
            x_transformed2 = x[:, spiral_dims[1]-1:]
            x_transformed = self.transform(x_transformed1, x_transformed2)
            return x_transformed, y

    def mutual_information(self) -> float:
        """Return de joint mutual information."""
        return self._base_sampler.mutual_information()
