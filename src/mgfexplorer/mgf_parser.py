"""
MGF (Mascot Generic Format) file parser for mass spectrometry data.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import re


class Spectrum:
    """Represents a single MS/MS spectrum from an MGF file."""

    def __init__(self, spectrum_id: Union[int, str]):
        self.spectrum_id = spectrum_id
        self.metadata: Dict[str, str] = {}
        self.ions: np.ndarray = np.array([])  # 2D array: [mz, intensity]

    def add_metadata(self, key: str, value: str):
        """Add metadata key-value pair, handling duplicate keys."""
        original_key = key
        counter = 1

        while key in self.metadata:
            key = f"{original_key}_{counter}"
            counter += 1

        self.metadata[key] = value

    def set_ions(self, mz_values: List[float], intensity_values: List[float]):
        """Set the ion data as a 2D numpy array."""
        if len(mz_values) != len(intensity_values):
            raise ValueError("mz and intensity arrays must have the same length")

        self.ions = np.column_stack((mz_values, intensity_values))

    def get_metadata_value(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        return self.metadata.get(key)

    def update_metadata(self, key: str, value: str):
        """Update existing metadata value or add new key-value pair."""
        self.metadata[key] = value

    def rename_metadata_key(self, old_key: str, new_key: str):
        """Rename a metadata key."""
        if old_key in self.metadata and new_key not in self.metadata:
            self.metadata[new_key] = self.metadata.pop(old_key)


class MGFParser:
    """Parser for MGF (Mascot Generic Format) files."""

    def __init__(self):
        self.spectra: List[Spectrum] = []

    def parse_file(self, file_path: str) -> List[Spectrum]:
        """Parse an MGF file and return a list of Spectrum objects."""
        self.spectra = []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        # Split content by BEGIN IONS blocks
        blocks = re.split(r"BEGIN IONS\s*\n", content)

        # Skip the first block (before any BEGIN IONS)
        for i, block in enumerate(blocks[1:], 1):
            if "END IONS" in block:
                spectrum = self._parse_spectrum_block(block, i)
                if spectrum:
                    self.spectra.append(spectrum)

        return self.spectra

    def parse_and_append_file(
        self, file_path: str, id_offset: int = 0
    ) -> List[Spectrum]:
        """Parse an MGF file and append spectra to existing list with ID offset."""
        new_spectra = []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        # Split content by BEGIN IONS blocks
        blocks = re.split(r"BEGIN IONS\s*\n", content)

        # Skip the first block (before any BEGIN IONS)
        for i, block in enumerate(blocks[1:], 1):
            if "END IONS" in block:
                spectrum = self._parse_spectrum_block(block, i + id_offset)
                if spectrum:
                    new_spectra.append(spectrum)
                    self.spectra.append(spectrum)

        return new_spectra

    def _parse_spectrum_block(self, block: str, spectrum_id: int) -> Optional[Spectrum]:
        """Parse a single spectrum block."""
        lines = block.split("\n")
        spectrum = Spectrum(spectrum_id)

        mz_values = []
        intensity_values = []

        for line in lines:
            line = line.strip()

            if not line or line == "END IONS":
                continue

            # Check if line contains key=value metadata
            if "=" in line and not self._is_ion_line(line):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    spectrum.add_metadata(key, value)

            # Check if line contains ion data (mz intensity)
            elif self._is_ion_line(line):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        mz_values.append(mz)
                        intensity_values.append(intensity)
                    except ValueError:
                        continue

        # Set ion data
        if mz_values and intensity_values:
            spectrum.set_ions(mz_values, intensity_values)

        return spectrum if spectrum.metadata or len(mz_values) > 0 else None

    def _is_ion_line(self, line: str) -> bool:
        """Check if a line contains ion data (mz intensity)."""
        parts = line.split()
        if len(parts) >= 2:
            try:
                float(parts[0])  # mz
                float(parts[1])  # intensity
                return True
            except ValueError:
                return False
        return False

    def get_all_metadata_keys(self) -> List[str]:
        """Get all unique metadata keys across all spectra."""
        all_keys = set()
        for spectrum in self.spectra:
            all_keys.update(spectrum.metadata.keys())
        return sorted(list(all_keys))

    def get_unique_values_for_key(self, key: str) -> List[str]:
        """Get all unique values for a specific metadata key, including empty string for missing values."""
        values = set()
        has_missing = False

        for spectrum in self.spectra:
            if key in spectrum.metadata:
                values.add(spectrum.metadata[key])
            else:
                has_missing = True

        # Include empty string if any spectrum is missing this key
        if has_missing:
            values.add("")

        return sorted(list(values))

    def get_spectrum_by_id(self, spectrum_id: Union[int, str]) -> Optional[Spectrum]:
        """Get a spectrum by its ID."""
        for spectrum in self.spectra:
            if spectrum.spectrum_id == spectrum_id:
                return spectrum
        return None

    def get_spectra_by_ids(self, spectrum_ids: List[Union[int, str]]) -> List[Spectrum]:
        """Get multiple spectra by their IDs."""
        result = []
        for spectrum_id in spectrum_ids:
            spectrum = self.get_spectrum_by_id(spectrum_id)
            if spectrum:
                result.append(spectrum)
        return result

    def rename_key_in_all_spectra(self, old_key: str, new_key: str):
        """Rename a metadata key in all spectra."""
        for spectrum in self.spectra:
            spectrum.rename_metadata_key(old_key, new_key)

    def update_key_value_in_selected_spectra(
        self, spectrum_ids: List[Union[int, str]], key: str, value: str
    ):
        """Update a metadata value in selected spectra."""
        for spectrum in self.spectra:
            if spectrum.spectrum_id in spectrum_ids:
                spectrum.update_metadata(key, value)

    def add_missing_key_to_all_spectra(self, key: str, default_value: str = ""):
        """Add a key with default value to all spectra that don't have it."""
        for spectrum in self.spectra:
            if key not in spectrum.metadata:
                spectrum.add_metadata(key, default_value)

    def calculate_cosine_similarity(
        self, spectrum1: Spectrum, spectrum2: Spectrum, mz_tolerance: float = 0.1
    ) -> float:
        """
        Calculate cosine similarity between two spectra.

        Args:
            spectrum1: First spectrum
            spectrum2: Second spectrum
            mz_tolerance: m/z tolerance for peak matching

        Returns:
            Cosine similarity score (0-1)
        """
        if spectrum1.ions.size == 0 or spectrum2.ions.size == 0:
            return 0.0

        # Get m/z and intensity values
        mz1, int1 = spectrum1.ions[:, 0], spectrum1.ions[:, 1]
        mz2, int2 = spectrum2.ions[:, 0], spectrum2.ions[:, 1]

        # Normalize intensities to maximum abundant peak to account for scaling differences
        max_int1 = np.max(int1)
        max_int2 = np.max(int2)

        if max_int1 == 0 or max_int2 == 0:
            return 0.0

        int1 = int1 / max_int1
        int2 = int2 / max_int2

        # Early exit if no overlap possible
        if mz1.max() + mz_tolerance < mz2.min() or mz2.max() + mz_tolerance < mz1.min():
            return 0.0

        # Sort spectra by m/z for efficient matching (if not already sorted)
        if not np.all(mz1[:-1] <= mz1[1:]):
            sort_idx1 = np.argsort(mz1)
            mz1, int1 = mz1[sort_idx1], int1[sort_idx1]

        if not np.all(mz2[:-1] <= mz2[1:]):
            sort_idx2 = np.argsort(mz2)
            mz2, int2 = mz2[sort_idx2], int2[sort_idx2]

        # Pre-normalize intensities to unit vectors
        norm1 = np.sqrt(np.sum(int1**2))
        norm2 = np.sqrt(np.sum(int2**2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        int1_norm = int1 / norm1
        int2_norm = int2 / norm2

        # Use vectorized approach for peak matching
        # For each peak in spectrum1, find the closest peak in spectrum2 within tolerance
        dot_product = 0.0

        # Use searchsorted for efficient range finding
        j_start = 0
        for i, mz in enumerate(mz1):
            # Find the range of peaks in spectrum2 that could match
            left_bound = mz - mz_tolerance
            right_bound = mz + mz_tolerance

            # Use searchsorted to find the range efficiently
            left_idx = np.searchsorted(mz2[j_start:], left_bound, side="left") + j_start
            right_idx = (
                np.searchsorted(mz2[j_start:], right_bound, side="right") + j_start
            )

            if left_idx < right_idx and left_idx < len(mz2):
                # Find the closest peak within the range
                candidates = mz2[left_idx:right_idx]
                if len(candidates) > 0:
                    closest_idx = left_idx + np.argmin(np.abs(candidates - mz))
                    dot_product += int1_norm[i] * int2_norm[closest_idx]

                    # Optimization: update j_start to avoid re-searching earlier peaks
                    j_start = max(j_start, left_idx)

        return max(0.0, min(1.0, dot_product))  # Clamp to [0, 1]

    def calculate_similarity_matrix(
        self, spectrum_ids: List[Union[int, str]], mz_tolerance: float = 0.1
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix for selected spectra.

        Args:
            spectrum_ids: List of spectrum IDs to compare
            mz_tolerance: m/z tolerance for peak matching

        Returns:
            Symmetric similarity matrix
        """
        spectra = [s for s in self.spectra if s.spectrum_id in spectrum_ids]
        n = len(spectra)

        if n < 2:
            return np.array([[1.0]] if n == 1 else [])

        similarity_matrix = np.zeros((n, n))

        # Fill diagonal with 1.0
        for i in range(n):
            similarity_matrix[i, i] = 1.0

        # Calculate upper triangle only (since matrix is symmetric)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_cosine_similarity(
                    spectra[i], spectra[j], mz_tolerance
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric

        return similarity_matrix

    def calculate_similarity_matrix_batch(
        self,
        spectrum_ids: List[Union[int, str]],
        mz_tolerance: float = 0.1,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix with progress reporting.

        Args:
            spectrum_ids: List of spectrum IDs to compare
            mz_tolerance: m/z tolerance for peak matching
            progress_callback: Optional callback function for progress updates

        Returns:
            Symmetric similarity matrix
        """
        spectra = [s for s in self.spectra if s.spectrum_id in spectrum_ids]
        n = len(spectra)

        if n < 2:
            return np.array([[1.0]] if n == 1 else [])

        similarity_matrix = np.zeros((n, n))
        total_comparisons = n * (n - 1) // 2
        completed = 0

        # Fill diagonal with 1.0
        for i in range(n):
            similarity_matrix[i, i] = 1.0

        # Calculate upper triangle only
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_cosine_similarity(
                    spectra[i], spectra[j], mz_tolerance
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric

                completed += 1

                # Report progress
                if (
                    progress_callback
                    and completed % max(1, total_comparisons // 50) == 0
                ):
                    progress_callback(completed, total_comparisons)

        return similarity_matrix

    def export_to_mgf(
        self, file_path: str, spectrum_ids: Optional[List[Union[int, str]]] = None
    ):
        """
        Export spectra to an MGF file.

        Args:
            file_path: Path to the output MGF file
            spectrum_ids: Optional list of spectrum IDs to export. If None, exports all spectra.
        """
        # Determine which spectra to export
        if spectrum_ids is None:
            spectra_to_export = self.spectra
        else:
            spectra_to_export = [
                s for s in self.spectra if s.spectrum_id in spectrum_ids
            ]

        with open(file_path, "w", encoding="utf-8") as f:
            for spectrum in spectra_to_export:
                f.write("BEGIN IONS\n")

                # Write metadata
                for key, value in spectrum.metadata.items():
                    f.write(f"{key}={value}\n")

                # Write ion data
                if spectrum.ions.size > 0:
                    for mz, intensity in spectrum.ions:
                        f.write(f"{mz:.6f} {intensity:.6f}\n")

                f.write("END IONS\n\n")

    def normalize_intensities(
        self,
        spectrum_ids: Optional[List[Union[int, str]]] = None,
        mode: str = "max",
        scale: float = 1.0,
    ):
        """
        Normalize intensities in spectra.

        Args:
            spectrum_ids: Optional list of spectrum IDs to normalize. If None, normalizes all spectra.
            mode: "max" to normalize relative to most abundant peak, "sum" to normalize relative to total sum
            scale: target scale (1.0 for 0-1, 100.0 for 0-100, 1000.0 for 0-1000)
        """
        # Determine which spectra to normalize
        if spectrum_ids is None:
            spectra_to_normalize = self.spectra
        else:
            spectra_to_normalize = [
                s for s in self.spectra if s.spectrum_id in spectrum_ids
            ]

        for spectrum in spectra_to_normalize:
            if spectrum.ions.size > 0:
                # Get current intensities
                intensities = spectrum.ions[:, 1]

                if mode == "max":
                    # Normalize relative to maximum intensity
                    max_intensity = np.max(intensities)

                    # Only normalize if max intensity is not zero
                    if max_intensity > 0:
                        normalized_intensities = (intensities / max_intensity) * scale
                        spectrum.ions[:, 1] = normalized_intensities

                elif mode == "sum":
                    # Normalize relative to total sum of intensities
                    total_intensity = np.sum(intensities)

                    # Only normalize if total intensity is not zero
                    if total_intensity > 0:
                        normalized_intensities = (intensities / total_intensity) * scale
                        spectrum.ions[:, 1] = normalized_intensities

                else:
                    raise ValueError(
                        f"Unknown normalization mode: {mode}. Use 'max' or 'sum'."
                    )
