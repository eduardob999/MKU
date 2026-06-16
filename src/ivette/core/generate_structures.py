#!/usr/bin/env python3
"""
generate_structures.py

Ivette Structure Generator

Hierarchy:

Topology
    ↓
Scaffold
    ↓
Structure

A topology is a ring atom sequence.
A scaffold is a validated aromatic heterocycle.
A structure is a nitro-substituted scaffold.

Output:
    List[dict]
        {
            "SMILES": ...,
            "RingSize": ...,
            "HeteroatomCount": ...
        }
"""

from itertools import combinations
from datetime import datetime
from pathlib import Path
import json

from rdkit import Chem


VALID_RING_ATOMS = ["c", "n", "o", "s"]


# ─────────────────────────────────────────────
# Topology feasibility
# ─────────────────────────────────────────────

def feasible_topology(
    partial_topology: list,
    ring_size: int
) -> bool:
    """
    Conservative pruning for topology generation.
    """

    remaining = ring_size - len(partial_topology)

    carbon_count = partial_topology.count("c")
    nitrogen_count = partial_topology.count("n")
    oxygen_count = partial_topology.count("o")
    sulfur_count = partial_topology.count("s")

    if carbon_count + remaining < 2:
        return False

    if len(partial_topology) >= 2:
        a, b = partial_topology[-2], partial_topology[-1]

        if {a, b} in (
            {"o", "o"},
            {"s", "s"},
            {"o", "s"},
        ):
            return False

    if ring_size == 5:

        if nitrogen_count > 2:
            return False

        if oxygen_count > 1:
            return False

        if sulfur_count > 1:
            return False

    if ring_size == 6:

        if nitrogen_count > 3:
            return False

    return True


def topology_closure_valid(
    topology: list
) -> bool:
    """
    Validate final ring closure adjacency.
    """

    if len(topology) < 2:
        return True

    a = topology[-1]
    b = topology[0]

    if {a, b} in (
        {"o", "o"},
        {"s", "s"},
        {"o", "s"},
    ):
        return False

    return True


def max_consecutive_nitrogens(
    topology: list
) -> int:
    """
    Longest nitrogen run in a cyclic topology.
    """

    doubled = topology + topology

    best = 0
    run = 0

    for atom in doubled:

        if atom == "n":
            run += 1
        else:
            run = 0

        best = max(best, run)

    return min(best, len(topology))


# ─────────────────────────────────────────────
# Topology symmetry reduction
# ─────────────────────────────────────────────

def canonical_topology(
    topology: list
) -> tuple:
    """
    Canonical representation under rotation
    and reflection symmetry.
    """

    n = len(topology)

    forward = [
        tuple(topology[i:] + topology[:i])
        for i in range(n)
    ]

    reversed_topology = list(reversed(topology))

    backward = [
        tuple(reversed_topology[i:] + reversed_topology[:i])
        for i in range(n)
    ]

    return min(forward + backward)


# ─────────────────────────────────────────────
# Scaffold generation
# ─────────────────────────────────────────────

def build_scaffold(
    topology: list,
    nh_positions: set
):
    """
    Construct aromatic scaffold from topology.
    """

    rw = Chem.RWMol()

    atom_indices = []

    for index, atom_symbol in enumerate(topology):

        atom = Chem.Atom(atom_symbol.upper())
        atom.SetIsAromatic(True)

        if atom_symbol == "n":
            atom.SetNumExplicitHs(
                1 if index in nh_positions else 0
            )

        atom_indices.append(
            rw.AddAtom(atom)
        )

    ring_size = len(atom_indices)

    for i in range(ring_size):

        rw.AddBond(
            atom_indices[i],
            atom_indices[(i + 1) % ring_size],
            Chem.BondType.AROMATIC,
        )

    scaffold = rw.GetMol()

    try:
        Chem.SanitizeMol(scaffold)
        return scaffold

    except Exception:
        return None


def enumerate_scaffolds(
    topology: list
):
    """
    Generate all chemically valid scaffolds
    for a topology.
    """

    nitrogen_positions = [
        i
        for i, atom in enumerate(topology)
        if atom == "n"
    ]

    nh_counts = [0, 1] if len(topology) == 5 else [0]

    scaffolds = []
    seen_scaffold_smiles = set()

    for nh_count in nh_counts:

        for nh_positions in combinations(
            nitrogen_positions,
            nh_count
        ):

            scaffold = build_scaffold(
                topology,
                set(nh_positions)
            )

            if scaffold is None:
                continue

            smiles = Chem.MolToSmiles(
                scaffold,
                canonical=True,
                isomericSmiles=False,
            )

            if smiles in seen_scaffold_smiles:
                continue

            seen_scaffold_smiles.add(smiles)
            scaffolds.append(scaffold)

    return scaffolds


# ─────────────────────────────────────────────
# Structure generation
# ─────────────────────────────────────────────

def substitution_sites(
    scaffold
):
    """
    Aromatic carbon atoms available for nitro substitution.
    """

    return [
        atom.GetIdx()
        for atom in scaffold.GetAtoms()
        if atom.GetIsAromatic()
        and atom.GetAtomicNum() == 6
    ]


def build_structure(
    scaffold,
    carbon_site: int
):
    """
    Attach a nitro group to a scaffold.
    """

    rw = Chem.RWMol(scaffold)

    nitrogen_index = rw.AddAtom(Chem.Atom("N"))
    oxygen1_index = rw.AddAtom(Chem.Atom("O"))
    oxygen2_index = rw.AddAtom(Chem.Atom("O"))

    rw.GetAtomWithIdx(
        nitrogen_index
    ).SetFormalCharge(+1)

    rw.GetAtomWithIdx(
        oxygen2_index
    ).SetFormalCharge(-1)

    rw.AddBond(
        carbon_site,
        nitrogen_index,
        Chem.BondType.SINGLE,
    )

    rw.AddBond(
        nitrogen_index,
        oxygen1_index,
        Chem.BondType.DOUBLE,
    )

    rw.AddBond(
        nitrogen_index,
        oxygen2_index,
        Chem.BondType.SINGLE,
    )

    try:
        structure = rw.GetMol()
        Chem.SanitizeMol(structure)
        return structure

    except Exception:
        return None


# ─────────────────────────────────────────────
# Topology exploration
# ─────────────────────────────────────────────

def explore_topologies(
    ring_size: int,
    partial_topology: list,
    structures: list,
    seen_structures: set,
    seen_topologies: set,
):

    if len(partial_topology) == ring_size:

        if not topology_closure_valid(
            partial_topology
        ):
            return

        if max_consecutive_nitrogens(
            partial_topology
        ) > 3:
            return

        topology = canonical_topology(
            partial_topology
        )

        if topology in seen_topologies:
            return

        seen_topologies.add(topology)

        heteroatom_count = sum(
            atom != "c"
            for atom in partial_topology
        )

        for scaffold in enumerate_scaffolds(
            partial_topology
        ):

            for site in substitution_sites(
                scaffold
            ):

                structure = build_structure(
                    scaffold,
                    site
                )

                if structure is None:
                    continue

                smiles = Chem.MolToSmiles(
                    structure,
                    canonical=True,
                    isomericSmiles=False,
                )

                if smiles in seen_structures:
                    continue

                seen_structures.add(smiles)

                structures.append(
                    {
                        "SMILES": smiles,
                        "RingSize": ring_size,
                        "HeteroatomCount": heteroatom_count,
                    }
                )

        return

    for atom in VALID_RING_ATOMS:

        next_topology = (
            partial_topology + [atom]
        )

        if feasible_topology(
            next_topology,
            ring_size
        ):
            explore_topologies(
                ring_size,
                next_topology,
                structures,
                seen_structures,
                seen_topologies,
            )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_structures(
    ring_sizes=(5, 6),
    set_id=None,
):
    """
    Main Ivette entry point.

    Generates a StructureSet.

    Returns:

    {
        "set_id": "...",
        "created": "...",
        "metadata": {...},
        "structures": [...]
    }

    """

    if set_id is None:
        set_id = (
            f"set_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )


    structures = []

    global_seen_structures = set()


    for ring_size in ring_sizes:

        seen_topologies = set()


        explore_topologies(
            ring_size=ring_size,
            partial_topology=[],
            structures=structures,
            seen_structures=global_seen_structures,
            seen_topologies=seen_topologies,
        )


    structures.sort(
        key=lambda row: (
            row["RingSize"],
            row["HeteroatomCount"],
            row["SMILES"],
        )
    )


    # Assign Ivette structure IDs

    for index, structure in enumerate(structures):

        structure["structure_id"] = (
            f"str_{index:06d}"
        )


    structure_set = {

        "set_id": set_id,

        "created": (
            datetime.now()
            .isoformat(timespec="seconds")
        ),

        "metadata": {

            "generator": "generate_structures.py",

            "ring_sizes": list(ring_sizes),

            "allowed_atoms": VALID_RING_ATOMS,

            "structure_count": len(structures),

        },

        "structures": structures,

    }


    return structure_set