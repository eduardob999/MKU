#!/usr/bin/env python3
"""Ivette CLI entry point.

The interactive workflow is organized as:

    Structure libraries -> data/structures/structure_*.csv  (metadata.json)
    Compound libraries  -> data/compounds/compound_*.csv     (metadata.json)
    Property datasets   -> data/datasets/runs/dataset_*/     (metadata.json)
    Models              -> data/models/runs/model_*/         (metadata.json)
    Geometry sets       -> data/geometries/runs/geometry_*/  (metadata.json)

Persistence lives in :mod:`ivette.util.storage`, session state in
:mod:`ivette.cli.context`, and the menus in :mod:`ivette.cli.menus`.
"""

from ivette.cli.menus import main


if __name__ == "__main__":
    main()
