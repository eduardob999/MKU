"""Launch a report figure: ``python -m ivette.viz KIND [IDS...] [--save PNG]``.

Used both standalone and as a subprocess spawned by the CLI's Results explorer.
Opens an interactive window by default; ``--save`` renders a PNG headlessly.
"""

import argparse
import sys

import matplotlib

# Choose the backend before importing pyplot (via plots): Agg for --save.
if "--save" in sys.argv:
    matplotlib.use("Agg")
else:
    for _backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue

from matplotlib import pyplot as plt  # noqa: E402

from ivette.viz import plots  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ivette report figures")
    parser.add_argument("kind", choices=sorted(plots.BUILDERS))
    parser.add_argument("ids", nargs="*", help="Entity id(s) for the figure")
    parser.add_argument("--save", default=None, help="Render to this PNG instead of showing")
    args = parser.parse_args(argv)

    fig, _keep = plots.BUILDERS[args.kind](*args.ids)

    if args.save:
        fig.savefig(args.save, dpi=120, facecolor=fig.get_facecolor())
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
