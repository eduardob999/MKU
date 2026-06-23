#!/usr/bin/env python3
"""Ivette CLI entry point.

The interactive workflow is organized as:

    Structure Sets  ->  data/structure/set_*.csv      (metadata.json)
    Compound Sets   ->  data/compounds/cset_*.csv      (metadata.json)
    Thermo Runs     ->  data/thermo/runs/run_*/        (metadata.json)
    Models          ->  data/models/runs/model_*/      (metadata.json)
    SDF Sets        ->  data/sdfs/runs/sdf_*/          (metadata.json)

Persistence lives in :mod:`ivette.util.storage`, session state in
:mod:`ivette.cli.context`, and the menus in :mod:`ivette.cli.menus`.
"""

from ivette.cli.menus import main


if __name__ == "__main__":
    main()
