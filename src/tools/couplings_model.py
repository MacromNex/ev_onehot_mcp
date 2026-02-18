"""
EVmutation Couplings Model Tools

This MCP Server provides 4 tools for analyzing evolutionary couplings and mutation effects:
1. evmutation_load_model: Load EVmutation parameters from plmc output file
2. evmutation_calculate_mutation_effects: Calculate Hamiltonian changes for mutations
3. evmutation_compute_couplings: Extract evolutionary coupling scores (CN, FN, MI)
4. evmutation_visualize_landscape: Visualize single mutant energy landscape

All tools extracted from combining-evolutionary-and-assay-labelled-data/src/couplings_model.py.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
from collections.abc import Iterable
from copy import deepcopy
from numba import jit
import matplotlib.pyplot as plt

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("COUPLINGS_MODEL_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("COUPLINGS_MODEL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
couplings_model_mcp = FastMCP(name="couplings_model")

# Configure matplotlib
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Constants
_SLICE = np.s_[:]
HAMILTONIAN_COMPONENTS = [FULL, COUPLINGS, FIELDS] = [0, 1, 2]
NUM_COMPONENTS = len(HAMILTONIAN_COMPONENTS)


# Methods for fast calculations (numba jit compiled)
@jit(nopython=True)
def _hamiltonians(sequences, J_ij, h_i):
    """
    Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
    for a given sequence A_1,...,A_L from J_ij and h_i parameters

    Parameters
    ----------
    sequences : np.array
        Sequence matrix for which Hamiltonians will be computed
    J_ij: np.array
        L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix

    Returns
    -------
    np.array
        Float matrix of size len(sequences) x 3, where each row corresponds to the
        1) total Hamiltonian of sequence and the 2) J_ij and 3) h_i sub-sums
    """
    N, L = sequences.shape
    H = np.zeros((N, NUM_COMPONENTS))
    for s in range(N):
        A = sequences[s]
        hi_sum = 0.0
        Jij_sum = 0.0
        for i in range(L):
            hi_sum += h_i[i, A[i]]
            for j in range(i + 1, L):
                Jij_sum += J_ij[i, j, A[i], A[j]]

        H[s] = [Jij_sum + hi_sum, Jij_sum, hi_sum]

    return H


@jit(nopython=True)
def _single_mutant_hamiltonians(target_seq, J_ij, h_i):
    """
    Calculate matrix of all possible single-site substitutions

    Parameters
    ----------
    target_seq : np.array(int)
        Target sequence for which mutant energy differences will be calculated
    J_ij: np.array
        L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix

    Returns
    -------
    np.array
        Float matrix of size L x num_symbols x 3, where the first two dimensions correspond to
        Hamiltonian differences compared to target sequence for all possible substitutions in
        all positions, and the third dimension corresponds to the deltas of
        1) total Hamiltonian and the 2) J_ij and 3) h_i sub-sums
    """
    L, num_symbols = h_i.shape
    H = np.empty((L, num_symbols, NUM_COMPONENTS))

    for i in range(L):
        for A_i in range(num_symbols):
            delta_hi = h_i[i, A_i] - h_i[i, target_seq[i]]
            delta_Jij = 0.0

            for j in range(L):
                if i != j:
                    delta_Jij += (
                        J_ij[i, j, A_i, target_seq[j]] -
                        J_ij[i, j, target_seq[i], target_seq[j]]
                    )

            H[i, A_i] = [delta_Jij + delta_hi, delta_Jij, delta_hi]

    return H


@jit(nopython=True)
def _delta_hamiltonian(pos, subs, target_seq, J_ij, h_i):
    """
    Parameters
    ----------
    pos : np.array(int)
        Vector of substituted positions
    subs : np.array(int)
        Vector of symbols above positions are substituted to
    target_seq : np.array(int)
        Target sequence for which mutant energy differences will be calculated
        relative to
    J_ij: np.array
        L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix

    Returns
    -------
    np.array
        Vector of length 3, where elements correspond to delta of
        1) total Hamiltonian and the 2) J_ij and 3) h_i sub-sums
    """
    L, num_symbols = h_i.shape

    M = pos.shape[0]
    delta_hi = 0.0
    delta_Jij = 0.0

    for m in range(M):
        i = pos[m]
        A_i = subs[m]

        delta_hi += h_i[i, A_i] - h_i[i, target_seq[i]]

        for j in range(L):
            if i != j:
                delta_Jij += (
                    J_ij[i, j, A_i, target_seq[j]] -
                    J_ij[i, j, target_seq[i], target_seq[j]]
                )

        for n in range(m + 1, M):
            j = pos[n]
            A_j = subs[n]
            delta_Jij -= J_ij[i, j, A_i, target_seq[j]]
            delta_Jij -= J_ij[i, j, target_seq[i], A_j]
            delta_Jij += J_ij[i, j, target_seq[i], target_seq[j]]
            delta_Jij += J_ij[i, j, A_i, A_j]

    return np.array([delta_Jij + delta_hi, delta_Jij, delta_hi])


class CouplingsModel:
    """
    Class to store parameters of pairwise undirected graphical model of sequences
    and compute evolutionary couplings, sequence statistical energies, etc.
    """

    def __init__(self, filename, precision="float32", file_format="plmc_v2", **kwargs):
        """
        Initializes the object with raw values read from binary .Jij file

        Parameters
        ----------
        filename : str
            Binary Jij file containing model parameters from plmc software
        precision : {"float32", "float64"}, default: "float32"
            Sets if input file has single (float32) or double precision (float64)
        file_format : {"plmc_v2", "plmc_v1"}, default: "plmc_v2"
            File format of parameter file
        """
        if file_format == "plmc_v2":
            self.__read_plmc_v2(filename, precision)
        elif file_format == "plmc_v1":
            self.__read_plmc_v1(
                filename, precision, kwargs.get("alphabet", None)
            )
        else:
            raise ValueError(
                "Illegal file format {}, valid options are: "
                "plmc_v2, plmc_v1".format(file_format)
            )

        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        try:
            self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
            self.has_target_seq = (np.sum(self.target_seq_mapped) > 0)
        except KeyError:
            self.target_seq_mapped = np.zeros((self.L), dtype=np.int32)
            self.has_target_seq = False

        self._reset_precomputed()

    def _reset_precomputed(self):
        """Delete precomputed values"""
        self._single_mut_mat_full = None
        self._double_mut_mat = None
        self._cn_scores = None
        self._fn_scores = None
        self._mi_scores_raw = None
        self._mi_scores_apc = None
        self._ecs = None

    def __read_plmc_v2(self, filename, precision):
        """Read updated Jij file format from plmc."""
        with open(filename, "rb") as f:
            self.L, self.num_symbols, self.N_valid, self.N_invalid, self.num_iter = (
                np.fromfile(f, "int32", 5)
            )

            self.theta, self.lambda_h, self.lambda_J, self.lambda_group, self.N_eff = (
                np.fromfile(f, precision, 5)
            )

            self.alphabet = np.fromfile(
                f, "S1", self.num_symbols
            ).astype("U1")

            self.weights = np.fromfile(
                f, precision, self.N_valid + self.N_invalid
            )

            self._target_seq = np.fromfile(f, "S1", self.L).astype("U1")
            self.index_list = np.fromfile(f, "int32", self.L)

            self.f_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            self.h_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            self.f_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            self.J_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.f_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.f_ij[j, i] = self.f_ij[i, j].T

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.J_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.J_ij[j, i] = self.J_ij[i, j].T

    def __read_plmc_v1(self, filename, precision, alphabet=None):
        """Read original eij/Jij file format from plmc."""
        GAP = "-"
        ALPHABET_PROTEIN_NOGAP = "ACDEFGHIKLMNPQRSTVWY"
        ALPHABET_PROTEIN = GAP + ALPHABET_PROTEIN_NOGAP

        with open(filename, "rb") as f:
            self.L, = np.fromfile(f, "int32", 1)
            self.num_symbols, = np.fromfile(f, "int32", 1)

            if alphabet is None:
                if self.num_symbols == 21:
                    alphabet = ALPHABET_PROTEIN
                elif self.num_symbols == 20:
                    alphabet = ALPHABET_PROTEIN_NOGAP
                else:
                    raise ValueError(
                        "Could not guess default alphabet for "
                        "{} states, specify alphabet parameter.".format(
                            self.num_symbols
                        )
                    )
            else:
                if len(alphabet) != self.num_symbols:
                    raise ValueError(
                        "Size of alphabet ({}) does not agree with "
                        "number of states in model ({})".format(
                            len(alphabet), self.num_symbols
                        )
                    )

            self.alphabet = np.array(list(alphabet))

            self._target_seq = np.fromfile(f, "S1", self.L).astype("U1")
            self.index_list = np.fromfile(f, "int32", self.L)

            self.N_valid = None
            self.N_invalid = None
            self.num_iter = None
            self.theta = None
            self.lambda_h = None
            self.lambda_J = None
            self.lambda_group = None
            self.N_eff = None
            self.weights = None

            self.f_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            self.h_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            self.f_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            self.J_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    file_i, file_j = np.fromfile(f, "int32", 2)

                    if i + 1 != file_i or j + 1 != file_j:
                        raise ValueError(
                            "Error: column pair indices inconsistent. "
                            "Expected: {} {}; File: {} {}".format(i + 1, j + 1, file_i, file_j)
                        )

                    self.f_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.f_ij[j, i] = self.f_ij[i, j].T

                    self.J_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.J_ij[j, i] = self.J_ij[i, j].T

    @property
    def target_seq(self):
        """Target/Focus sequence of model"""
        return self._target_seq

    @target_seq.setter
    def target_seq(self, sequence):
        """Define a new target sequence"""
        self._reset_precomputed()

        if len(sequence) != self.L:
            raise ValueError(
                "Sequence length inconsistent with model length: {} {}".format(
                    len(sequence), self.L
                )
            )

        if isinstance(sequence, str):
            sequence = list(sequence)

        self._target_seq = np.array(sequence)
        self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
        self.has_target_seq = True

    @property
    def index_list(self):
        """Sequence indices mapping"""
        return self._index_list

    @index_list.setter
    def index_list(self, mapping):
        """Define a new number mapping for sequences"""
        if len(mapping) != self.L:
            raise ValueError(
                "Mapping length inconsistent with model length: {} {}".format(
                    len(mapping), self.L
                )
            )

        self._index_list = np.array(mapping)
        self.index_map = {b: a for a, b in enumerate(self.index_list)}

    def convert_sequences(self, sequences):
        """Convert sequences to internal symbol representation"""
        seq_lens = list(set(map(len, sequences)))
        if len(seq_lens) != 1:
            raise ValueError("Input sequences have different lengths: " + str(seq_lens))

        L_seq = seq_lens[0]
        if L_seq != self.L:
            raise ValueError(
                "Sequence lengths do not correspond to model length: {} {}".format(
                    L_seq, self.L
                )
            )

        S = np.empty((len(sequences), L_seq), dtype=np.int)

        try:
            for i, s in enumerate(sequences):
                S[i] = [self.alphabet_map[x] for x in s]
        except KeyError:
            raise ValueError("Invalid symbol in sequence {}: {}".format(i, x))

        return S

    def hamiltonians(self, sequences):
        """Calculate Hamiltonians for given sequences"""
        if isinstance(sequences, list):
            sequences = self.convert_sequences(sequences)

        return _hamiltonians(sequences, self.J_ij, self.h_i)

    @property
    def single_mut_mat_full(self):
        """Hamiltonian difference for all possible single-site variants"""
        if self._single_mut_mat_full is None:
            self._single_mut_mat_full = _single_mutant_hamiltonians(
                self.target_seq_mapped, self.J_ij, self.h_i
            )

        return self._single_mut_mat_full

    @property
    def single_mut_mat(self):
        """Hamiltonian difference for all possible single-site variants"""
        return self.single_mut_mat_full[:, :, FULL]

    def delta_hamiltonian(self, substitutions, verify_mutants=True):
        """Calculate difference in statistical energy relative to target sequence"""
        pos = np.empty(len(substitutions), dtype=np.int32)
        subs = np.empty(len(substitutions), dtype=np.int32)

        try:
            for i, (subs_pos, subs_from, subs_to) in enumerate(substitutions):
                pos[i] = self.index_map[subs_pos]
                subs[i] = self.alphabet_map[subs_to]
                if verify_mutants and subs_from != self.target_seq[pos[i]]:
                    raise ValueError(
                        "Inconsistency with target sequence: pos={} target={} subs={}".format(
                            subs_pos, self.target_seq[i], subs_from
                        )
                    )
        except KeyError:
            raise ValueError(
                "Illegal substitution: {}{}{}\nAlphabet: {}\nPositions: {}".format(
                    subs_from, subs_pos, subs_to, self.alphabet_map, self.index_list
                )
            )

        return _delta_hamiltonian(pos, subs, self.target_seq_mapped, self.J_ij, self.h_i)

    @property
    def double_mut_mat(self):
        """Hamiltonian difference for all possible double mutant variants"""
        if self._double_mut_mat is None:
            self._double_mut_mat = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            seq = self.target_seq_mapped
            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self._double_mut_mat[i, j] = (
                        np.tile(self.single_mut_mat[i], (self.num_symbols, 1)).T +
                        np.tile(self.single_mut_mat[j], (self.num_symbols, 1)) +
                        self.J_ij[i, j] -
                        np.tile(self.J_ij[i, j, :, seq[j]], (self.num_symbols, 1)).T -
                        np.tile(self.J_ij[i, j, seq[i], :], (self.num_symbols, 1)) +
                        self.J_ij[i, j, seq[i], seq[j]])

                    self._double_mut_mat[j, i] = self._double_mut_mat[i, j].T

        return self._double_mut_mat

    @classmethod
    def apc(cls, matrix):
        """Apply average product correction (APC) to matrix"""
        L = matrix.shape[0]
        if L != matrix.shape[1]:
            raise ValueError("Input matrix is not symmetric: {}".format(matrix.shape))

        col_means = np.mean(matrix, axis=0) * L / (L - 1)
        matrix_mean = np.mean(matrix) * L / (L - 1)

        apc = np.dot(
            col_means.reshape(L, 1), col_means.reshape(1, L)
        ) / matrix_mean

        corrected_matrix = matrix - apc
        corrected_matrix[np.diag_indices(L)] = 0

        return corrected_matrix

    def _calculate_ecs(self):
        """Calculate FN and CN scores and MI scores"""
        self._fn_scores = np.zeros((self.L, self.L))
        self._mi_scores_raw = np.zeros((self.L, self.L))

        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                self._fn_scores[i, j] = np.linalg.norm(self.J_ij[i, j], "fro")
                self._fn_scores[j, i] = self._fn_scores[i, j]

                p = self.f_ij[i, j]
                m = np.dot(self.f_i[i, np.newaxis].T, self.f_i[j, np.newaxis])
                self._mi_scores_raw[i, j] = np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
                self._mi_scores_raw[j, i] = self._mi_scores_raw[i, j]

        self._cn_scores = self.apc(self._fn_scores)
        self._mi_scores_apc = self.apc(self._mi_scores_raw)

        ecs = []
        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                ecs.append((
                    self.index_list[i], self.target_seq[i],
                    self.index_list[j], self.target_seq[j],
                    abs(self.index_list[i] - self.index_list[j]),
                    self._mi_scores_raw[i, j], self._mi_scores_apc[i, j],
                    self._fn_scores[i, j], self._cn_scores[i, j]
                ))

        self._ecs = pd.DataFrame(
            ecs, columns=["i", "A_i", "j", "A_j", "seqdist", "mi_raw", "mi_apc", "fn", "cn"]
        ).sort_values(by="cn", ascending=False)

    @property
    def cn_scores(self):
        """L x L numpy matrix with CN (corrected norm) scores"""
        if self._cn_scores is None:
            self._calculate_ecs()
        return self._cn_scores

    @property
    def fn_scores(self):
        """L x L numpy matrix with FN (Frobenius norm) scores"""
        if self._fn_scores is None:
            self._calculate_ecs()
        return self._fn_scores

    @property
    def mi_scores_raw(self):
        """L x L numpy matrix with MI scores without APC correction"""
        if self._mi_scores_raw is None:
            self._calculate_ecs()
        return self._mi_scores_raw

    @property
    def mi_scores_apc(self):
        """L x L numpy matrix with MI scores with APC correction"""
        if self._mi_scores_apc is None:
            self._calculate_ecs()
        return self._mi_scores_apc

    @property
    def ecs(self):
        """DataFrame with evolutionary couplings, sorted by CN score"""
        if self._ecs is None:
            self._calculate_ecs()
        return self._ecs


@couplings_model_mcp.tool
def evmutation_load_model(
    model_path: Annotated[str, "Path to EVmutation model file (.params or .model_params) from plmc software"] = None,
    file_format: Annotated[Literal["plmc_v2", "plmc_v1"], "Model file format"] = "plmc_v2",
    precision: Annotated[Literal["float32", "float64"], "Numerical precision of parameters"] = "float32",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Load EVmutation parameters from plmc software output file for evolutionary coupling analysis.
    Input is binary model file from plmc and output is model summary and target sequence information.
    """
    # Input validation
    if model_path is None:
        raise ValueError("Path to EVmutation model file must be provided")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = CouplingsModel(str(model_file), precision=precision, file_format=file_format)

    # Generate output prefix
    if out_prefix is None:
        out_prefix = f"evmutation_model_{timestamp}"

    # Save model summary
    summary_file = OUTPUT_DIR / f"{out_prefix}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("EVmutation Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model length (L): {model.L}\n")
        f.write(f"Number of symbols: {model.num_symbols}\n")
        f.write(f"Alphabet: {''.join(model.alphabet)}\n")
        f.write(f"Target sequence: {''.join(model.target_seq)}\n")
        f.write(f"Index list: {model.index_list.tolist()}\n\n")

        if file_format == "plmc_v2":
            f.write(f"Valid sequences: {model.N_valid}\n")
            f.write(f"Invalid sequences: {model.N_invalid}\n")
            f.write(f"Iterations: {model.num_iter}\n")
            f.write(f"Theta: {model.theta}\n")
            f.write(f"Lambda_h: {model.lambda_h}\n")
            f.write(f"Lambda_J: {model.lambda_J}\n")
            f.write(f"Lambda_group: {model.lambda_group}\n")
            f.write(f"N_eff: {model.N_eff}\n")

    # Save target sequence
    target_seq_file = OUTPUT_DIR / f"{out_prefix}_target_sequence.txt"
    with open(target_seq_file, "w") as f:
        f.write("".join(model.target_seq))

    return {
        "message": f"Loaded EVmutation model: L={model.L}, alphabet={''.join(model.alphabet)}",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/couplings_model.py",
        "artifacts": [
            {
                "description": "Model summary",
                "path": str(summary_file.resolve())
            },
            {
                "description": "Target sequence",
                "path": str(target_seq_file.resolve())
            }
        ]
    }


@couplings_model_mcp.tool
def evmutation_calculate_mutation_effects(
    model_path: Annotated[str, "Path to EVmutation model file from plmc software"] = None,
    mutations: Annotated[str, "Comma-separated mutations in format pos:from:to (e.g., '10:A:V,20:L:F')"] = None,
    file_format: Annotated[Literal["plmc_v2", "plmc_v1"], "Model file format"] = "plmc_v2",
    precision: Annotated[Literal["float32", "float64"], "Numerical precision"] = "float32",
    verify_mutants: Annotated[bool, "Verify mutation 'from' matches target sequence"] = True,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Calculate Hamiltonian (statistical energy) changes for specified mutations using EVmutation model.
    Input is model file and mutation specifications and output is delta Hamiltonian values for each mutation.
    """
    # Input validation
    if model_path is None:
        raise ValueError("Path to EVmutation model file must be provided")
    if mutations is None:
        raise ValueError("Mutations must be provided in format 'pos:from:to,pos:from:to'")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Parse mutations
    mutation_list = []
    for mut_str in mutations.split(","):
        mut_str = mut_str.strip()
        parts = mut_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid mutation format: {mut_str}. Expected 'pos:from:to'")
        pos, from_aa, to_aa = int(parts[0]), parts[1], parts[2]
        mutation_list.append((pos, from_aa, to_aa))

    # Load model
    model = CouplingsModel(str(model_file), precision=precision, file_format=file_format)

    # Calculate mutation effects
    delta_H = model.delta_hamiltonian(mutation_list, verify_mutants=verify_mutants)

    # Generate output prefix
    if out_prefix is None:
        out_prefix = f"mutation_effects_{timestamp}"

    # Save results
    results_file = OUTPUT_DIR / f"{out_prefix}_results.csv"
    results_df = pd.DataFrame({
        "mutations": [mutations],
        "delta_H_total": [delta_H[0]],
        "delta_H_couplings": [delta_H[1]],
        "delta_H_fields": [delta_H[2]]
    })
    results_df.to_csv(results_file, index=False)

    # Save detailed report
    report_file = OUTPUT_DIR / f"{out_prefix}_report.txt"
    with open(report_file, "w") as f:
        f.write("EVmutation Mutation Effects\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Target sequence: {''.join(model.target_seq)}\n\n")
        f.write("Mutations:\n")
        for pos, from_aa, to_aa in mutation_list:
            f.write(f"  {from_aa}{pos}{to_aa}\n")
        f.write("\nResults:\n")
        f.write(f"  Total ΔH: {delta_H[0]:.4f}\n")
        f.write(f"  Couplings (ΔJ_ij): {delta_H[1]:.4f}\n")
        f.write(f"  Fields (Δh_i): {delta_H[2]:.4f}\n")

    return {
        "message": f"Calculated mutation effects: ΔH_total={delta_H[0]:.4f}",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/couplings_model.py",
        "artifacts": [
            {
                "description": "Mutation effects results",
                "path": str(results_file.resolve())
            },
            {
                "description": "Detailed report",
                "path": str(report_file.resolve())
            }
        ]
    }


@couplings_model_mcp.tool
def evmutation_compute_couplings(
    model_path: Annotated[str, "Path to EVmutation model file from plmc software"] = None,
    file_format: Annotated[Literal["plmc_v2", "plmc_v1"], "Model file format"] = "plmc_v2",
    precision: Annotated[Literal["float32", "float64"], "Numerical precision"] = "float32",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Compute evolutionary coupling scores (CN, FN, MI) from EVmutation model parameters.
    Input is model file from plmc and output is evolutionary coupling scores sorted by CN.
    """
    # Input validation
    if model_path is None:
        raise ValueError("Path to EVmutation model file must be provided")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = CouplingsModel(str(model_file), precision=precision, file_format=file_format)

    # Generate output prefix
    if out_prefix is None:
        out_prefix = f"evolutionary_couplings_{timestamp}"

    # Compute coupling scores
    ecs_df = model.ecs

    # Save evolutionary couplings
    ecs_file = OUTPUT_DIR / f"{out_prefix}_couplings.csv"
    ecs_df.to_csv(ecs_file, index=False)

    # Save CN score matrix
    cn_matrix_file = OUTPUT_DIR / f"{out_prefix}_cn_matrix.csv"
    cn_df = pd.DataFrame(
        model.cn_scores,
        index=model.index_list,
        columns=model.index_list
    )
    cn_df.to_csv(cn_matrix_file)

    # Save summary statistics
    summary_file = OUTPUT_DIR / f"{out_prefix}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Evolutionary Couplings Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model length: {model.L}\n")
        f.write(f"Number of couplings: {len(ecs_df)}\n\n")
        f.write("Top 10 couplings (by CN score):\n")
        f.write(ecs_df.head(10).to_string(index=False))
        f.write("\n\nCoupling score statistics:\n")
        f.write(ecs_df[["mi_raw", "mi_apc", "fn", "cn"]].describe().to_string())

    return {
        "message": f"Computed {len(ecs_df)} evolutionary couplings",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/couplings_model.py",
        "artifacts": [
            {
                "description": "Evolutionary couplings table",
                "path": str(ecs_file.resolve())
            },
            {
                "description": "CN score matrix",
                "path": str(cn_matrix_file.resolve())
            },
            {
                "description": "Summary statistics",
                "path": str(summary_file.resolve())
            }
        ]
    }


@couplings_model_mcp.tool
def evmutation_visualize_landscape(
    model_path: Annotated[str, "Path to EVmutation model file from plmc software"] = None,
    file_format: Annotated[Literal["plmc_v2", "plmc_v1"], "Model file format"] = "plmc_v2",
    precision: Annotated[Literal["float32", "float64"], "Numerical precision"] = "float32",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize single mutant energy landscape from EVmutation model showing all possible substitutions.
    Input is model file from plmc and output is heatmap and distribution plots of mutation effects.
    """
    # Input validation
    if model_path is None:
        raise ValueError("Path to EVmutation model file must be provided")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = CouplingsModel(str(model_file), precision=precision, file_format=file_format)

    # Generate output prefix
    if out_prefix is None:
        out_prefix = f"mutant_landscape_{timestamp}"

    # Get single mutant matrix
    single_mut_mat = model.single_mut_mat

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of all single mutant effects
    im1 = axes[0].imshow(single_mut_mat, cmap='RdBu_r', aspect='auto')
    axes[0].set_xlabel('Amino Acid', fontsize=12)
    axes[0].set_ylabel('Position', fontsize=12)
    axes[0].set_title('Single Mutant Energy Landscape (ΔH)', fontsize=14)
    axes[0].set_xticks(range(model.num_symbols))
    axes[0].set_xticklabels(model.alphabet)
    plt.colorbar(im1, ax=axes[0], label='ΔH')

    # Distribution of mutant effects
    all_effects = single_mut_mat.flatten()
    axes[1].hist(all_effects, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
    axes[1].set_xlabel('Mutant Effect (ΔH)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Single Mutant Effects', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    figure_file = OUTPUT_DIR / f"{out_prefix}_landscape.png"
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Save single mutant matrix
    matrix_file = OUTPUT_DIR / f"{out_prefix}_matrix.csv"
    matrix_df = pd.DataFrame(
        single_mut_mat,
        index=model.index_list,
        columns=model.alphabet
    )
    matrix_df.to_csv(matrix_file)

    return {
        "message": f"Visualized {model.L} positions × {model.num_symbols} substitutions",
        "reference": "https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/master/src/couplings_model.py",
        "artifacts": [
            {
                "description": "Mutational landscape visualization",
                "path": str(figure_file.resolve())
            },
            {
                "description": "Single mutant effect matrix",
                "path": str(matrix_file.resolve())
            }
        ]
    }
