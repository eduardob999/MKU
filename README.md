Download physicochemical data by substructure
===========================================

This repository includes a lightweight tool to query PubChem for compounds
matching a SMILES substructure and download physicochemical properties.

Files added:
- [scripts/download_physchem.py](scripts/download_physchem.py)
- [requirements.txt](requirements.txt)

-----------

1. Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the script for a single substructure (e.g., benzene ring):

```bash
python scripts/download_physchem.py --smiles "c1ccccc1" --max 200 --output benzene.csv
```

3. Or provide an input file with one SMILES per line:

```bash
python scripts/download_physchem.py --input-file subs.smi --max 100 --output subs_results.csv
```

4. Generate nitro-substituted aromatic monocycle substructure patterns:

```bash
python scripts/generate_nitro_aromatic_substructures.py --ring-size 6 --atoms c n o s --output nitro_aromatics.txt
```

Generate multiple ring sizes at once:

```bash
python scripts/generate_nitro_aromatic_substructures.py --ring-size 5 6 7 --atoms c n o s --output nitro_multiple.txt
```

For an interactive CLI menu, run:

```bash
python scripts/generate_nitro_aromatic_substructures.py --menu
```

The script also supports 5- and 7-membered aromatic rings, including heterocycles like imidazole and diazepine variants, and it filters combinations to preserve reasonable Hückel aromaticity.

You can also include pyrrole-like nitrogen explicitly with `nh` when needed:

```bash
python scripts/generate_nitro_aromatic_substructures.py --ring-size 5 --atoms c n nh o s --output nitro_5rings.txt
```

To print a generic matching pattern instead of enumerating every ring SMILES:

```bash
python scripts/generate_nitro_aromatic_substructures.py --generic --ring-size 6 --atoms c n o s
```

Notes and next steps
--------------------
- The current implementation queries PubChem only. It can be extended to
  fetch from ChEMBL or other databases by mapping InChIKey/CID to external IDs.
- For SMARTS queries, consider converting SMARTS to an example SMILES or
  performing local substructure filtering with RDKit if you have a molecule set.
