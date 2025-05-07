"""Command line interface for my-chai-extension."""

import logging
import typer
from pathlib import Path

from fksfold.chai_fks import run_inference
from chai_lab.data.parsing.msas.aligned_pqt import merge_a3m_in_directory

logging.basicConfig(level=logging.INFO)

CITATION = """
@article{FKSFold-Technical-Report,
	title        = {FKSFold: Improving AlphaFold3-Type Predictions of Molecular Glue-Induced Ternary Complexes with Feynman-Kac-Steered Diffusion},
	author       = {Shen, Jian and Zhou, Shengmin and Che, Xing},
	year         = 2025,
	journal      = {bioRxiv},
	publisher    = {Cold Spring Harbor Laboratory},
	doi          = {10.1101/2025.05.03.651455},
	url          = {https://www.biorxiv.org/content/10.1101/2025.05.03.651455v1},
	elocation-id = {2025.05.03.651455},
	eprint       = {https://www.biorxiv.org/content/10.1101/2025.05.03.651455v1.full.pdf}
}

@article{Chai-1-Technical-Report,
	title        = {Chai-1: Decoding the molecular interactions of life},
	author       = {{Chai Discovery}},
	year         = 2024,
	journal      = {bioRxiv},
	publisher    = {Cold Spring Harbor Laboratory},
	doi          = {10.1101/2024.10.10.615955},
	url          = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955},
	elocation-id = {2024.10.10.615955},
	eprint       = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955.full.pdf}
}
""".strip()


def citation():
    """Print citation information"""
    typer.echo(CITATION)


def cli():
    app = typer.Typer()
    app.command("fold", help="Run FKSFold to fold a complex.")(run_inference)
    app.command(
        "a3m-to-pqt",
        help="Convert all a3m files in a directory for a *single sequence* into a aligned parquet file",
    )(merge_a3m_in_directory)
    # app.command("citation", help="Print citation information")(citation)
    app()


if __name__ == "__main__":
    cli()
