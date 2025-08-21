# Wuzhengxiao 2025/7/27


import os
import time
import numpy as np


class Molecule:
    def __init__(
        self,
        position,
        key_points,
        site_states=np.array([0, 0, 0, 0]),
        orientation=0,
        molecule_type="unknown",
        is_manipulated=False,
        operated_time=0,
        spectroscopy_points=None,
        spectroscopy_completed=False,
        spectroscopy_time=0,
        spectroscopy_points_count=0,
        successful_measurements=0,
        spectroscopy_results=None,
    ):
        self.position = position  # molecule position (x, y)
        self.key_points = key_points  # molecule key points
        self.site_states = site_states  # molecule site states
        self.orientation = orientation  # molecule orientation, default is 0
        self.registration_time = time.time()  # registration time, using current time
        self.molecule_type = (
            molecule_type  # type/kind of molecule (e.g., "TPM", "Br", "H")
        )
        self.is_manipulated = is_manipulated  # whether molecule is manipulated
        self.operated_time = operated_time  # operated time, default is 0
        # Spectroscopy-specific attributes
        self.spectroscopy_points = (
            spectroscopy_points or []
        )  # list of spectroscopy points
        self.spectroscopy_completed = spectroscopy_completed  # spectroscopy done?
        self.spectroscopy_time = spectroscopy_time  # spectroscopy timestamp
        self.spectroscopy_points_count = spectroscopy_points_count  # number of points
        self.successful_measurements = (
            successful_measurements  # count of successful measurements
        )
        self.spectroscopy_results = spectroscopy_results or []  # list of result dicts

    def __repr__(self):
        return (
            f"Molecule(position={self.position}, type={self.molecule_type}, "
            f"manipulated={self.is_manipulated}, orientation={self.orientation}, "
            f"spectroscopy_completed={self.spectroscopy_completed}, "
            f"spectroscopy_points_count={self.spectroscopy_points_count})"
        )

    def mark_spectroscopy_completed(self, points, results, spectroscopy_time=None):
        """Mark spectroscopy as completed and store results."""
        self.spectroscopy_points = points
        self.spectroscopy_points_count = len(points)
        self.spectroscopy_results = results
        self.spectroscopy_completed = True
        self.spectroscopy_time = spectroscopy_time or time.time()
        self.successful_measurements = sum(1 for r in results if r.get("success"))

    def get_spectroscopy_success_rate(self):
        """Get the success rate of spectroscopy measurements for this molecule."""
        if self.spectroscopy_points_count == 0:
            return 0.0
        return self.successful_measurements / self.spectroscopy_points_count

    def is_spectroscopy_candidate(self):
        """Check if molecule is ready for spectroscopy (manipulated but not measured)."""
        return self.is_manipulated and not self.spectroscopy_completed

    def get_spectroscopy_summary(self):
        """Get a summary dict of spectroscopy data for this molecule."""
        return {
            "position": self.position,
            "molecule_type": self.molecule_type,
            "is_manipulated": self.is_manipulated,
            "spectroscopy_completed": self.spectroscopy_completed,
            "points_count": self.spectroscopy_points_count,
            "successful_measurements": self.successful_measurements,
            "success_rate": self.get_spectroscopy_success_rate(),
            "spectroscopy_time": self.spectroscopy_time,
        }


class Registry:
    def __init__(self):
        self.molecules = []
        self.threshold = 0.5

    def register_molecule(
        self,
        position,
        key_points,
        site_states=np.array([0, 0, 0, 0]),
        orientation=0,
        molecule_type="unknown",
        is_manipulated=False,
        operated_time=0,
        spectroscopy_points=None,
        spectroscopy_completed=False,
        spectroscopy_time=0,
        spectroscopy_points_count=0,
        successful_measurements=0,
        spectroscopy_results=None,
        fresh_all_position=True,
    ):
        new_molecule = Molecule(
            position,
            key_points,
            site_states,
            orientation,
            molecule_type,
            is_manipulated,
            operated_time,
            spectroscopy_points,
            spectroscopy_completed,
            spectroscopy_time,
            spectroscopy_points_count,
            successful_measurements,
            spectroscopy_results,
        )

        # Check for existing similar molecules and handle drift correction
        drift_x_list = []
        drift_y_list = []

        for index, molecule in enumerate(self.molecules):
            distance = np.linalg.norm(
                np.array(molecule.position) * 10**9 - np.array(position) * 10**9
            )
            if distance < self.threshold:
                drift_x = position[0] - molecule.position[0]
                drift_y = position[1] - molecule.position[1]
                drift_x_list.append(drift_x)
                drift_y_list.append(drift_y)

        if drift_x_list and drift_y_list:
            median_drift_x = np.median(drift_x_list)
            median_drift_y = np.median(drift_y_list)

            if fresh_all_position:
                self.move_all_molecules(median_drift_x, median_drift_y)

            # Replace existing molecule
            for index, molecule in enumerate(self.molecules):
                distance = np.linalg.norm(
                    np.array(molecule.position) * 10**9 - np.array(position) * 10**9
                )
                if distance < self.threshold:
                    self.molecules[index] = new_molecule
                    return index

        # Register new molecule if no existing one found
        self.molecules.append(new_molecule)
        return len(self.molecules) - 1

    def get_all_molecules(self):
        return self.molecules

    def move_all_molecules(self, delta_x, delta_y):
        """Move all molecules by the given delta."""
        for molecule in self.molecules:
            molecule.position = (
                molecule.position[0] + delta_x,
                molecule.position[1] + delta_y,
            )
            molecule.key_points = [
                (x + delta_x, y + delta_y) for x, y in molecule.key_points
            ]

    def update_molecule(self, index, **kwargs):
        """Update attributes of a molecule."""
        molecule = self.molecules[index]
        for key, value in kwargs.items():
            if hasattr(molecule, key):
                setattr(molecule, key, value)

    def mark_spectroscopy_completed(
        self, index, points, results, spectroscopy_time=None
    ):
        """Mark a molecule as having completed spectroscopy."""
        molecule = self.molecules[index]
        molecule.mark_spectroscopy_completed(points, results, spectroscopy_time)

    def get_spectroscopy_candidates(self):
        """Get molecules ready for spectroscopy."""
        candidates = []
        for index, molecule in enumerate(self.molecules):
            if molecule.is_manipulated and not molecule.spectroscopy_completed:
                candidates.append((molecule, index))
        return candidates

    def get_spectroscopy_statistics(self):
        """Get spectroscopy statistics."""
        total_molecules = len(self.molecules)
        manipulated_molecules = sum(1 for m in self.molecules if m.is_manipulated)
        spectroscopy_completed = sum(
            1 for m in self.molecules if m.spectroscopy_completed
        )
        total_points = sum(m.spectroscopy_points_count for m in self.molecules)
        total_successful = sum(m.successful_measurements for m in self.molecules)

        return {
            "total_molecules": total_molecules,
            "manipulated_molecules": manipulated_molecules,
            "spectroscopy_completed": spectroscopy_completed,
            "total_spectroscopy_points": total_points,
            "successful_measurements": total_successful,
            "spectroscopy_completion_rate": (
                spectroscopy_completed / manipulated_molecules
                if manipulated_molecules > 0
                else 0
            ),
            "measurement_success_rate": (
                total_successful / total_points if total_points > 0 else 0
            ),
        }

    def export_spectroscopy_summary(self, filepath):
        """Export spectroscopy summary to JSON file."""
        import json

        summary = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": self.get_spectroscopy_statistics(),
            "molecules": [],
        }

        for index, molecule in enumerate(self.molecules):
            mol_data = {
                "index": index,
                "position": molecule.position,
                "orientation": molecule.orientation,
                "molecule_type": molecule.molecule_type,
                "is_manipulated": molecule.is_manipulated,
                "operated_time": molecule.operated_time,
                "spectroscopy_completed": molecule.spectroscopy_completed,
                "spectroscopy_time": molecule.spectroscopy_time,
                "spectroscopy_points_count": molecule.spectroscopy_points_count,
                "successful_measurements": molecule.successful_measurements,
                "spectroscopy_points": molecule.spectroscopy_points,
                "spectroscopy_results": molecule.spectroscopy_results,
            }
            summary["molecules"].append(mol_data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def find_molecules_by_spectroscopy_status(
        self, completed=None, minimum_success_rate=None
    ):
        """Find molecules based on spectroscopy criteria."""
        matches = []

        for index, molecule in enumerate(self.molecules):
            if completed is not None and molecule.spectroscopy_completed != completed:
                continue

            if (
                minimum_success_rate is not None
                and molecule.spectroscopy_points_count > 0
            ):
                success_rate = (
                    molecule.successful_measurements
                    / molecule.spectroscopy_points_count
                )
                if success_rate < minimum_success_rate:
                    continue

            matches.append((molecule, index))

        return matches

    def find_closest_molecule(self, coord):
        """
        Find the closest molecule to given coordinates.
        """
        if not self.molecules:
            return None, -1

        min_distance = float("inf")
        closest_molecule = None
        closest_index = -1

        for index, molecule in enumerate(self.molecules):
            distance = np.linalg.norm(np.array(molecule.position) - np.array(coord))
            if distance < min_distance:
                min_distance = distance
                closest_molecule = molecule
                closest_index = index

        return closest_molecule, closest_index

    def find_first_unreacted_molecule(self):
        """Find the first unmanipulated molecule."""
        if not self.molecules:
            return None, -1

        for index, molecule in enumerate(self.molecules):
            if not molecule.is_manipulated:
                return molecule, index

        return None, -1

    def find_molecules_by_type(self, molecule_type):
        """Find molecules by type."""
        matches = []
        for index, molecule in enumerate(self.molecules):
            if molecule.molecule_type == molecule_type:
                matches.append((molecule, index))
        return matches

    def find_manipulated_molecules(self):
        """Find all manipulated molecules."""
        matches = []
        for index, molecule in enumerate(self.molecules):
            if molecule.is_manipulated:
                matches.append((molecule, index))
        return matches

    def clear_the_registry(self):
        """Clear the molecule registry."""
        self.molecules.clear()

    def __repr__(self):
        return f"Registry(molecules={self.molecules})"
