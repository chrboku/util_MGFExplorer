"""
Molecular formula utilities for fragment annotation.
"""

import re
import os
import pickle
import hashlib
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools


class FormulaCache:
    """Persistent cache for molecular formula calculations with batched writes."""

    def __init__(self, cache_dir=None, write_threshold=10000):
        """Initialize the cache with a specified directory."""
        if cache_dir is None:
            # Default cache directory in user's home/.mgfexplorer_cache
            cache_dir = os.path.expanduser("~/.mgfexplorer_cache")

        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "formula_cache.pkl")
        self._cache = {}
        self._new_entries_count = 0  # Track new entries since last save
        self._write_threshold = (
            write_threshold  # Number of new entries before auto-save
        )
        self._dirty = False  # Track if cache has unsaved changes
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk once at startup."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                print(f"Loaded formula cache with {len(self._cache)} entries")
            else:
                self._cache = {}
                print("No existing cache file found, starting with empty cache")
        except Exception as e:
            print(f"Warning: Could not load formula cache: {e}")
            self._cache = {}

    def _save_cache(self, force=False):
        """Save cache to disk only when threshold is reached or forced."""
        if not self._dirty and not force:
            return  # Nothing to save

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"Saved formula cache with {len(self._cache)} entries to disk")
            self._new_entries_count = 0
            self._dirty = False
        except Exception as e:
            print(f"Warning: Could not save formula cache: {e}")

    def _generate_key(
        self,
        precursor_formula: str,
        additional_elements: List[str],
        ppm_tolerance: float,
    ) -> str:
        """Generate a unique cache key for the given parameters."""
        # Create a deterministic key based on parameters
        key_data = f"{precursor_formula}|{sorted(additional_elements)}|{ppm_tolerance}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_subformulas(
        self,
        precursor_formula: str,
        additional_elements: List[str] = None,
        ppm_tolerance: float = 50.0,
    ):
        """Get cached subformulas or return None if not cached."""
        if additional_elements is None:
            additional_elements = []

        key = self._generate_key(precursor_formula, additional_elements, ppm_tolerance)
        return self._cache.get(key)

    def store_subformulas(
        self,
        precursor_formula: str,
        subformulas: List[Dict],
        additional_elements: List[str] = None,
        ppm_tolerance: float = 50.0,
    ):
        """Store subformulas in cache with batched writing."""
        if additional_elements is None:
            additional_elements = []

        key = self._generate_key(precursor_formula, additional_elements, ppm_tolerance)

        # Check if this is a new entry
        is_new_entry = key not in self._cache

        # Store with timestamp for potential cleanup
        cache_entry = {
            "subformulas": subformulas,
            "timestamp": time.time(),
            "precursor_formula": precursor_formula,
            "additional_elements": additional_elements,
            "ppm_tolerance": ppm_tolerance,
        }

        self._cache[key] = cache_entry
        self._dirty = True

        # Track new entries for batched writing
        if is_new_entry:
            self._new_entries_count += 1

            # Auto-save when threshold is reached
            if self._new_entries_count >= self._write_threshold:
                print(
                    f"Cache threshold reached ({self._new_entries_count} new entries), saving to disk..."
                )
                self._save_cache()

    def force_save(self):
        """Force save the cache to disk regardless of threshold."""
        self._save_cache(force=True)

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._new_entries_count = 0
        self._dirty = False
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            print("Cache file deleted from disk")

    def get_cache_stats(self):
        """Get cache statistics including pending writes."""
        return {
            "total_entries": len(self._cache),
            "new_entries_pending": self._new_entries_count,
            "write_threshold": self._write_threshold,
            "has_unsaved_changes": self._dirty,
            "cache_file": self.cache_file,
            "cache_size_mb": os.path.getsize(self.cache_file) / (1024 * 1024)
            if os.path.exists(self.cache_file)
            else 0,
        }


# Global cache instance
_formula_cache = FormulaCache()


class MolecularFormula:
    """Class for handling molecular formulas and their manipulation."""

    # Common atomic masses (monoisotopic)
    ATOMIC_MASSES = {
        "H": 1.007825032,
        "C": 12.000000000,
        "N": 14.003074005,
        "O": 15.994914620,
        "P": 30.973761998,
        "S": 31.972071174,
        "F": 18.998403163,
        "Cl": 34.968852682,
        "Br": 78.918337600,
        "I": 126.904473000,
        "Na": 22.989769282,
        "K": 38.963706679,
        "Ca": 39.962591012,
        "Mg": 23.985041697,
        "Fe": 55.934937475,
        "Zn": 63.929142222,
        "Cu": 62.929597474,
        "Mn": 54.938045141,
        "Co": 58.933195048,
        "Ni": 57.935342921,
        "Se": 79.916521271,
        "Si": 27.976926535,
        "Al": 26.981538627,
        "B": 11.009305360,
    }

    def __init__(self, formula: str = ""):
        """Initialize with a formula string."""
        self.composition = self._parse_formula(formula) if formula else defaultdict(int)

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """Parse a molecular formula string into element counts."""
        composition = defaultdict(int)

        # Remove whitespace and handle common formatting
        formula = formula.strip().replace(" ", "")

        # Pattern to match element and count: Element followed by optional number
        pattern = r"([A-Z][a-z]?)(\d*)"
        matches = re.findall(pattern, formula)

        for element, count_str in matches:
            count = int(count_str) if count_str else 1
            composition[element] += count

        return dict(composition)

    def get_exact_mass(self) -> float:
        """Calculate the exact mass of the molecular formula."""
        mass = 0.0
        for element, count in self.composition.items():
            if element in self.ATOMIC_MASSES:
                mass += self.ATOMIC_MASSES[element] * count
            else:
                # Unknown element, skip
                continue
        return mass

    def to_string(self) -> str:
        """Convert the composition back to a formula string."""
        if not self.composition:
            return ""

        # Sort elements in common order: C, H, then alphabetical
        elements = list(self.composition.keys())
        ordered_elements = []

        # Add C first if present
        if "C" in elements:
            ordered_elements.append("C")
            elements.remove("C")

        # Add H second if present
        if "H" in elements:
            ordered_elements.append("H")
            elements.remove("H")

        # Add remaining elements alphabetically
        ordered_elements.extend(sorted(elements))

        formula_parts = []
        for element in ordered_elements:
            count = self.composition[element]
            if count > 1:
                formula_parts.append(f"{element}{count}")
            else:
                formula_parts.append(element)

        return "".join(formula_parts)

    def subtract(self, other: "MolecularFormula") -> "MolecularFormula":
        """Subtract another formula from this one."""
        result = MolecularFormula()

        for element in self.composition:
            result.composition[element] = self.composition[element]

        for element, count in other.composition.items():
            result.composition[element] -= count
            if result.composition[element] <= 0:
                del result.composition[element]

        return result

    def add(self, other: "MolecularFormula") -> "MolecularFormula":
        """Add another formula to this one."""
        result = MolecularFormula()

        # Add all elements from both formulas
        all_elements = set(self.composition.keys()) | set(other.composition.keys())

        for element in all_elements:
            count = self.composition.get(element, 0) + other.composition.get(element, 0)
            if count > 0:
                result.composition[element] = count

        return result

    def is_subset_of(self, other: "MolecularFormula") -> bool:
        """Check if this formula is a subset of another formula."""
        for element, count in self.composition.items():
            if other.composition.get(element, 0) < count:
                return False
        return True

    def copy(self) -> "MolecularFormula":
        """Create a copy of this formula."""
        result = MolecularFormula()
        result.composition = dict(self.composition)
        return result


class FragmentAnnotator:
    """Class for annotating fragments with possible molecular formulas."""

    def __init__(
        self,
        precursor_formula: str,
        additional_elements: List[str] = None,
        ppm_tolerance: float = 50.0,
    ):
        """
        Initialize the fragment annotator.

        Args:
            precursor_formula: The molecular formula of the precursor ion
            additional_elements: Additional elements that can be added during fragmentation
            ppm_tolerance: Mass tolerance in ppm for matching
        """
        self.precursor_formula = MolecularFormula(precursor_formula)
        self.additional_elements = additional_elements or []
        self.ppm_tolerance = ppm_tolerance
        self._cached_subformulas = None  # Cache subformulas once generated

    def generate_subformulas(
        self, max_additional_elements: int = 3
    ) -> List[MolecularFormula]:
        """Generate all possible sub-formulas of the precursor."""
        # Check if we already have cached subformulas for this instance
        if self._cached_subformulas is not None:
            return self._cached_subformulas

        # Check persistent cache first
        precursor_formula_str = self.precursor_formula.to_string()
        cached_result = _formula_cache.get_subformulas(
            precursor_formula_str, self.additional_elements, self.ppm_tolerance
        )

        if cached_result is not None:
            # Convert cached string formulas back to MolecularFormula objects
            self._cached_subformulas = []
            for formula_data in cached_result["subformulas"]:
                formula = MolecularFormula(formula_data["formula_string"])
                self._cached_subformulas.append(formula)
            print(
                f"Loaded {len(self._cached_subformulas)} subformulas from cache for {precursor_formula_str}"
            )
            return self._cached_subformulas

        # Generate subformulas if not cached
        print(f"Generating subformulas for {precursor_formula_str}...")
        subformulas = []

        # Generate all combinations of the precursor elements
        base_subformulas = self._generate_element_combinations(self.precursor_formula)
        subformulas.extend(base_subformulas)

        # Add combinations with additional elements
        if self.additional_elements:
            additional_formula_options = []
            for element in self.additional_elements:
                for count in range(1, max_additional_elements + 1):
                    additional_formula = MolecularFormula()
                    additional_formula.composition[element] = count
                    additional_formula_options.append(additional_formula)

            # Combine base formulas with additional elements
            for base_formula in base_subformulas:
                for additional_formula in additional_formula_options:
                    combined = base_formula.add(additional_formula)
                    subformulas.append(combined)

        # Cache the results
        self._cached_subformulas = subformulas

        # Prepare data for persistent storage
        formula_data_for_cache = []
        for formula in subformulas:
            formula_data_for_cache.append(
                {
                    "formula_string": formula.to_string(),
                    "exact_mass": formula.get_exact_mass(),
                }
            )

        # Store in persistent cache
        _formula_cache.store_subformulas(
            precursor_formula_str,
            formula_data_for_cache,
            self.additional_elements,
            self.ppm_tolerance,
        )

        print(
            f"Generated and cached {len(subformulas)} subformulas for {precursor_formula_str}"
        )
        return subformulas

    def _generate_element_combinations(
        self, formula: MolecularFormula
    ) -> List[MolecularFormula]:
        """Generate all possible combinations of elements from the formula."""
        combinations = []

        elements = list(formula.composition.keys())
        counts = [formula.composition[element] for element in elements]

        # Generate all combinations using itertools
        count_ranges = [range(count + 1) for count in counts]

        for combination in itertools.product(*count_ranges):
            if sum(combination) == 0:  # Skip empty formula
                continue

            subformula = MolecularFormula()
            for i, element in enumerate(elements):
                if combination[i] > 0:
                    subformula.composition[element] = combination[i]

            combinations.append(subformula)

        return combinations

    def annotate_mz(self, mz_value: float, charge: int = 1) -> List[Dict[str, any]]:
        """
        Annotate a given m/z value with possible molecular formulas.

        Args:
            mz_value: The m/z value to annotate
            charge: The charge state (default: 1)

        Returns:
            List of annotation dictionaries containing formula and ppm error
        """
        # Calculate neutral mass
        neutral_mass = (mz_value * abs(charge)) - (
            charge * 1.007825032
        )  # Remove proton mass

        annotations = []
        subformulas = self.generate_subformulas()
        import time

        print(f"   Got {len(subformulas)} subformulas {time.time()}")

        for formula in (
            formula
            for formula in subformulas
            if abs(formula.get_exact_mass() - neutral_mass) / neutral_mass * 1e6
            <= self.ppm_tolerance
        ):
            # Calculate ppm error
            ppm_error = abs(
                (formula.get_exact_mass() - neutral_mass) / neutral_mass * 1e6
            )

            if ppm_error <= self.ppm_tolerance:
                annotations.append(
                    {
                        "formula": formula.to_string(),
                        "theoretical_mass": formula.get_exact_mass(),
                        "ppm_error": ppm_error,
                        "charge": charge,
                    }
                )

        # Sort by ppm error
        annotations.sort(key=lambda x: x["ppm_error"])

        return annotations

    def calculate_ppm_error(
        self, experimental_mass: float, theoretical_mass: float
    ) -> float:
        """Calculate ppm error between experimental and theoretical masses."""
        return abs((experimental_mass - theoretical_mass) / theoretical_mass * 1e6)


# Utility functions for cache management
def get_cache_stats() -> Dict:
    """Get cache statistics."""
    return _formula_cache.get_cache_stats()


def clear_formula_cache():
    """Clear the persistent formula cache."""
    _formula_cache.clear_cache()
    print("Formula cache cleared.")


def force_save_cache():
    """Force save the cache to disk regardless of threshold."""
    _formula_cache.force_save()
    print("Formula cache saved to disk.")


def get_cache_info() -> str:
    """Get human-readable cache information."""
    stats = get_cache_stats()
    pending_info = (
        f" ({stats['new_entries_pending']} pending)"
        if stats["new_entries_pending"] > 0
        else ""
    )
    return f"Cache: {stats['total_entries']} entries{pending_info}, {stats['cache_size_mb']:.1f} MB"
