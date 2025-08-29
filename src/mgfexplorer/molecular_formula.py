"""
Molecular formula utilities for fragment annotation.
"""

import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools


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
    ):
        """
        Initialize the fragment annotator.

        Args:
            precursor_formula: The molecular formula of the precursor ion
            additional_elements: Additional elements that can be added during fragmentation
        """
        self.precursor_formula = MolecularFormula(precursor_formula)
        self.additional_elements = additional_elements or []

    def generate_subformulas(
        self, max_additional_elements: int = 3
    ) -> List[MolecularFormula]:
        """Generate all possible sub-formulas of the precursor."""
        # Generate subformulas
        precursor_formula_str = self.precursor_formula.to_string()
        self.subformulas = []

        # Generate all combinations of the precursor elements
        base_subformulas = self._generate_element_combinations(self.precursor_formula)
        self.subformulas.extend(base_subformulas)

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
                    self.subformulas.append(combined)

        return self.subformulas

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

    def annotate_mz(
        self, mz_value: float, charge: int = 1, ppm_tolerance: float = 50.0
    ) -> List[Dict[str, any]]:
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

        for formula in (
            formula
            for formula in self.subformulas
            if abs(formula.get_exact_mass() - neutral_mass) / neutral_mass * 1e6
            <= ppm_tolerance
        ):
            # Calculate ppm error
            ppm_error = abs(
                (formula.get_exact_mass() - neutral_mass) / neutral_mass * 1e6
            )

            if ppm_error <= ppm_tolerance:
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
