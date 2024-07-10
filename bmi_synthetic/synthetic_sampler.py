import itertools
import numpy as np

from bmi.interface import KeyArray
from bmi.samplers.base import BaseSampler
from typing import Set, Tuple, Union, Dict, Any


def _check_parameters(
    cell_boundaries: np.ndarray,
    symbol_probabilities: np.ndarray,
    cell_probabilities: np.ndarray,
    ) -> None:
    """
    Checks the validity of the parameters: cell_boundaries, symbol_probabilities, and cell_probabilities.

    Parameters
    ----------
    cell_boundaries : np.ndarray
        The boundaries of the cells for each dimension.
    symbol_probabilities : np.ndarray
        The probabilities of each symbol.
    cell_probabilities : np.ndarray
        The probabilities of each cell for each symbol.

    Raises
    ------
    ValueError
        If any parameter does not meet the required conditions.
    """

    def _check_array_01_range(arr: np.ndarray) -> bool:
        # Check if all values in the array are in the range [0, 1]
        return np.all(a=(arr >= 0) & (arr <= 1))

    # Check if the number of symbols matches between symbol_probabilities and cell_probabilities
    if symbol_probabilities.shape[0] != cell_probabilities.shape[0]:
        raise ValueError("Cell probabilities must be defined for each symbol")
    # Check if the dimensionality matches between cell_boundaries and cell_probabilities
    if len(cell_boundaries) != cell_probabilities.ndim - 1:
        raise ValueError(
            "The dimensionality of the continuous component of the distribution must match between the arrays "
            "cell_boundaries and cell_probabilities"
        )
    # Check if the shape of cell_probabilities matches the intervals induced by cell_boundaries
    if (
        tuple(np.vectorize(pyfunc=lambda arr: len(arr) - 1)(cell_boundaries))
        != cell_probabilities.shape[1:]
    ):
                raise ValueError(
                    "The amount of intervals induced by cell_boundaries's arrays for each dimension must match the shape of "
                    "cell_probabilities after the symbol count"
        )
    # Check if each boundary array is sorted and contains no repeated values
    if not np.all(
        a=np.vectorize(pyfunc=lambda arr: np.all(arr[:-1] < arr[1:]))(cell_boundaries)
    ):
        raise ValueError(
            "The bound array for each dimension in cell_boundaries must be sorted and it must not contain repeated "
            "values"
        )
    # Check if all values in symbol_probabilities are in the range [0, 1]
    if not _check_array_01_range(arr=symbol_probabilities):
        raise ValueError("All values in symbol_probabilities must be in [0, 1]")
    # Check if all values in cell_probabilities are in the range [0, 1]
    if not _check_array_01_range(arr=cell_probabilities):
        raise ValueError("All values in cell_probabilities must be in [0, 1]")
    # Check if the sum of symbol probabilities is 1.0
    if np.sum(a=symbol_probabilities) != 1.0:
        raise ValueError(
            "The probabilities of all symbols in symbol_probabilities must sum 1.0"
        )
    # Check if the sum of cell probabilities for each symbol is 1.0
    if not np.all(
        a=np.isclose(
            np.sum(a=cell_probabilities, axis=tuple(range(1, cell_probabilities.ndim))),
            1.0,
            atol=1e-6,
        )
    ):
        raise ValueError(
            "The probabilities of all cells must sum 1.0 when conditioned to any possible symbol"
        )


class SyntheticSampler(BaseSampler):

    def __init__(
        self,
        cell_boundaries: np.ndarray,
        symbol_probabilities: np.ndarray,
        cell_probabilities: np.ndarray
        ) -> None:
        """
        Class constructor that initializes the distribution given its parameters.

        Parameters
        ----------
        cell_boundaries : np.ndarray
            Object array of numpy arrays such that "cell_boundaries[k][j]" is the cell boundary of index "j" for the
            dimension of index "k." The elements of cell_boundaries do not necessarily share the same length, i.e.,
            "cell_boundaries[k]" has shape of (n_{k+1}, ).
        symbol_probabilities : np.ndarray
            Array of shape (m, ), where "m" is the total number of symbols. The value of "symbol_probabilities[y]" is
            the probability for the symbol "y+1."
        cell_probabilities : np.ndarray
            Array of shape (m, n_1, n_2, ..., n_d) such that "cell_probabilities[y, i_1, i_2, ..., i_d]" is the
            conditioned probability for the cell whose dimension-wise lower bounds are "cell_boundaries[0][i_1]",
            "cell_boundaries[1][i_2]", ..., "cell_boundaries[d-1][i_d]" for the dimensions of index 0, 1, ..., d-1,
            respectively.

        Raises
        ------
        ValueError
            If there is any inconsistency within the distribution parameters.
        """

        _check_parameters(
            cell_boundaries=cell_boundaries,
            symbol_probabilities=symbol_probabilities,
            cell_probabilities=cell_probabilities,
        )

        self.cell_boundaries: np.ndarray = cell_boundaries
        self.symbol_probabilities: np.ndarray = symbol_probabilities
        self.cell_probabilities: np.ndarray = cell_probabilities

        # Define the dimension of the x variable
        dim_x: int = self.cell_probabilities.ndim - 1
        
        # Set the dimension of the y variable to 1
        super().__init__(dim_x=dim_x, dim_y=1)


    def sample(
        self,
        n_points: int,
        rng: Union[int, KeyArray],
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples data from the distribution

        Parameters
        ----------
        n_points : int
            Number of samples to generate
        rng : Union[int, KeyArray]
            Random generator's seed

        Returns
        -------
        A tuple (tup) such that tup[0] is an array of shape (N, d) and tup[1] is an array of shape (N, ) corresponding
        to the samples of X and Y, respectively; here "N" is the value of "n_samples" and "d" is the dimensionality of
        the continuous random variable X
        """
        rng: np.random.Generator = np.random.default_rng(
            seed=rng
            ) # Initialize a random number generator
        n_possible_symbols: int = len(
            self.symbol_probabilities
            ) # Get the number of possible symbols
        x_dimensionality: int = (
            self.cell_probabilities.ndim - 1
            ) # Determine the dimensionality of x

        # Generate an array of uniformly distributed random numbers for x variable
        x_array: np.ndarray = rng.uniform(
            low=0.0, high=1.0, size=(n_points, x_dimensionality)
            )
        
        # Generate an array of random symbols based on their probabilities for y variable
        y_array: np.ndarray = rng.choice(
            a=n_possible_symbols, size=n_points, p=self.symbol_probabilities
            )
        
        # Initialize an empty array to store cell indices
        cell_array: np.ndarry = np.empty(
            shape=(n_points, x_dimensionality)
            )

        # Get the unique y values that were sampled
        sampled_y_values: np.ndarray = np.unique(ar=y_array)
        
        # For each unique y value, fill the corresponding entries in cell_array
        y_value: int
        for y_value in sampled_y_values:
            # Create a mask for entries corresponding to the current y value
            mask: np.ndarray = y_array == y_value
            # Fill the entries for the current y value with randomly chosen cell indices based on their probabilities
            cell_array[mask] = rng.choice(
                a=np.array(
                    list(
                        itertools.product(
                            *[
                                np.arange(len(self.cell_boundaries[k]) - 1)
                                for k in range(x_dimensionality)
                            ]
                        )
                    )
                ),
                size=np.count_nonzero(mask),
                p=self.cell_probabilities[y_value].flatten(),
            )

        # Find the unique sampled cells and get the inverse indices
        sampled_cells: np.ndarray
        sampled_cells_inverse_idx: np.ndarray
        sampled_cells, sampled_cells_inverse_idx = np.unique(ar=cell_array, axis=0, return_inverse=True)
        
        # For each unique sampled cell, adjust the corresponding entries in x_array
        sampled_idx: int
        cell_indexes: np.ndarray
        for sampled_idx, cell_indexes in enumerate(sampled_cells):
            # Create a mask for entries corresponding to the current cell index
            mask: np.ndarray = sampled_cells_inverse_idx == sampled_idx
            # Get the lower and upper bounds of the current cell
            cell_low_bounds: np.ndarray = np.array(
                [
                    self.cell_boundaries[k][int(cell_indexes[k])]
                    for k in range(x_dimensionality)
                ]
            )
            cell_high_bounds: np.ndarray = np.array(
                [
                    self.cell_boundaries[k][int(cell_indexes[k] + 1)]
                    for k in range(x_dimensionality)
                ]
            )
            # Adjust the entries in x_array based on the bounds of the current cell
            x_array[mask] = (cell_high_bounds - cell_low_bounds) * x_array[
                mask
            ] + cell_low_bounds

        # Return the adjusted x_array and the y_array with 1 added to each element
        return x_array, y_array + 1


    def mutual_information(self) -> float:
        """
        Computes and returns the joint mutual information between $X$ and $Y$.

        Returns
        -------
        The analytical value of $I(X;Y)$
        """
        x_dimensionality: int = (
            self.cell_probabilities.ndim - 1
            ) # Determine the dimensionality of x
        coordinates = set(
            range(1, x_dimensionality + 1)
            )
        n_possible_symbols: int = len(
            self.symbol_probabilities
            ) # Get the number of possible symbols
        coord_array: np.ndarray = np.array(
            list(coordinates)
            ) # Convert the set of coordinates to a numpy array
        n_coordinates: int = len(
            coordinates
            ) # Get the number of coordinates provided
        
        # Sum the cell probabilities over the axes not in coordinates to get projected probabilities
        projected_probabilities: np.ndarray = np.sum(
            self.cell_probabilities,
            axis=tuple(set(range(1, x_dimensionality + 1)).difference(coordinates)),
        )
        # Reshape the symbol probabilities to match the shape of projected probabilities
        ex_sym_prob: np.ndarray = self.symbol_probabilities.reshape(
            (n_possible_symbols,) + (1,) * n_coordinates
        )
        # Compute the denominator of the logarithm term
        log_denominator: np.ndarray = np.sum(
            ex_sym_prob * projected_probabilities, axis=0
        )
        # We address these warning when ignoring the "nan" values in the overall summation
        with np.errstate(divide="ignore", invalid="ignore"):
            # Compute the inner terms of the mutual information formula
            inner_terms: np.ndarray = (
                ex_sym_prob
                * projected_probabilities
                * np.log2(projected_probabilities / log_denominator)
            )
        # Return the sum of the inner terms, ignoring any NaN values
        return float(np.nansum(inner_terms))


    def get_mutual_information(
        self,
        coordinates: Set[int],
        ) -> float:
        """
        Computes and returns the mutual information between $X_{\Lambda}$ and $Y$, where $\Lambda$ is the set of
        indexes defined by "coordinates".

        Parameters
        ----------
        coordinates : Set[int]
            Set of dimension coordinates of X to consider in the computation of the mutual information.

        Raises
        ------
        ValueError:
            If the coordinates are outside the set {1, 2, ..., d}.

        Returns
        -------
        The analytical value of $I(X_{\Lambda};Y)$
        """
        n_coordinates: int = len(
            coordinates
            )  # Get the number of coordinates provided
        n_possible_symbols: int = len(
            self.symbol_probabilities
            ) # Get the number of possible symbols
        x_dimensionality: int = (
            self.cell_probabilities.ndim - 1
            ) # Determine the dimensionality of x
        coord_array: np.ndarray = np.array(
            list(coordinates)
            ) # Convert the set of coordinates to a numpy array

        # Check if all coordinates are within the valid range
        if np.any((coord_array < 1) | (coord_array > x_dimensionality)):
            raise ValueError("All the coordinates must belong to {1, 2, ..., d}")

        # Sum the cell probabilities over the axes not in coordinates to get projected probabilities
        projected_probabilities: np.ndarray = np.sum(
            self.cell_probabilities,
            axis=tuple(set(range(1, x_dimensionality + 1)).difference(coordinates)),
            )

        # Reshape the symbol probabilities to match the shape of projected probabilities
        ex_sym_prob: np.ndarray = self.symbol_probabilities.reshape(
            (n_possible_symbols,) + (1,) * n_coordinates
            )

        # Compute the denominator of the logarithm term
        log_denominator: np.ndarray = np.sum(
            ex_sym_prob * projected_probabilities, axis=0
            )
        # We address these warning when ignoring the "nan" values in the overall summation
        with np.errstate(divide="ignore", invalid="ignore"):
            # Compute the inner terms of the mutual information formula
            inner_terms: np.ndarray = (
                ex_sym_prob
                * projected_probabilities
                * np.log2(projected_probabilities / log_denominator)
            )
        # Return the sum of the inner terms, ignoring any NaN values
        return float(np.nansum(inner_terms))


    def get_symbol_entropy(self) -> float:
        """
        Computes and returns the entropy of Y.

        Returns
        -------
        The analytical value of H(Y)
        """
        # Filter out zero probabilities from the symbol probabilities
        non_zero_prob: np.ndarray = self.symbol_probabilities[
            self.symbol_probabilities != 0.0
            ]
        # Compute the entropy using the non-zero probabilities
        return -np.sum(non_zero_prob * np.log2(non_zero_prob))


    def get_optimal_accuracy(self) -> float:
        """
        Computes the expected value for the accuracy of an optimal classificator for this distribution.

        Returns
        -------
        The expected accuracy for an optimal classificator.
        """
        return float(
            np.sum(
                np.max(
                    self.symbol_probabilities.reshape(
                        self.symbol_probabilities.shape
                        + (1,) * (self.cell_probabilities.ndim - 1)
                    )
                    * self.cell_probabilities,
                    axis=0,
                )
            )
        )


    def get_parameter_count(self) -> int:
        """
        Gets the total number of parameters used in describing this distribution.

        Returns
        -------
        Total number of parameters.
        """
        return (
            self.symbol_probabilities.size
            + sum(np.vectorize(pyfunc=lambda x: x.size)(self.cell_boundaries))
            + self.cell_probabilities.size
        )
        
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Gets the metadata dictionary of the distribution. Current keys are "entropy_y", "in_shape", "num_classes", and
        "max_acc1", which are the symbol entropy - in bits; H(Y) - the input dimensionality (d), the number of possible
        symbols (m) and the optimal (maximum achievable) accuracy. Additional keys are on the form "mi_[Lambda]" where
        "[Lambda]" is a hyphen-separated ordered sequence of values that conform Lambda, which are the MI value in bits
        between "\eta_{\Lambda}(X)" and "Y"; finally, the key "joint_mi" is the same as before, but when
        "\Lambda" = {1,2,...,d}.

        Returns
        -------
        The metadata dictionary.
        """
        x_dimensionality: int = (
            self.cell_probabilities.ndim - 1
            ) # Determine the dimensionality of x

        # Initialize the metadata dictionary
        metadata: Dict[str] = {
            "entropy_y": self.get_symbol_entropy(),  # Entropy of Y in bits
            "in_shape": [x_dimensionality],  # Input dimensionality (d)
            "num_classes": len(self.symbol_probabilities),  # Number of possible symbols (m)
            "max_acc1": self.get_optimal_accuracy(),  # Optimal (maximum achievable) accuracy
            "total_params": self.get_parameter_count(),  # Number of total parameters
            }

        # Update the metadata dictionary with mutual information values for all subsets of coordinates
        metadata.update(
            {
                f"mi_{'-'.join(map(str, sorted(set(lambda_set))))}": self.get_mutual_information(
                    coordinates=set(lambda_set)
                )
                for subset_size in range(1, x_dimensionality + 1)
                for lambda_set in itertools.combinations(
                    set(range(1, x_dimensionality + 1)), subset_size
                )
            }
        )

        # Add the joint mutual information for all coordinates
        metadata["joint_mi"] = self.mutual_information()

        return metadata

