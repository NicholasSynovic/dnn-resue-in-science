import sqlite3
from json import loads
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

DEFAULT_OUTPUT_PATH: Path = Path("figU.pdf").absolute()
SUPTITLE_FONT_SIZE: int = 24
TITLE_FONT_SIZE: int = 22
XY_LABEL_FONT_SIZE: int = 20
XY_TICK_FONT_SIZE: int = 18
OTHER_FONT_SIZE: int = XY_TICK_FONT_SIZE
SCIENTIFIC_WORKFLOW_STEPS: list[str] = [
    "Observation",
    "Hypothesis",
    "Background",
    "Test",
    "Analysis",
]
TOP_N: int = 5
FIGSIZE: tuple[float, float] = (25.6, 4.8)


def extract_steps(obj: Any) -> list[str]:
    """Extract flattened `step` values from a dict or list of dicts."""

    def _flatten_step(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []

        if isinstance(value, list):
            flattened: list[str] = []
            for item in value:
                flattened.extend(_flatten_step(item))
            return flattened

        if isinstance(value, dict):
            return _flatten_step(value.get("step")) if "step" in value else []

        return []

    if isinstance(obj, dict):
        return _flatten_step(obj.get("step"))

    if isinstance(obj, list):
        flattened: list[str] = []
        for item in obj:
            if isinstance(item, dict) and "step" in item:
                flattened.extend(_flatten_step(item.get("step")))
        return flattened

    return []


def normalize_data(df: DataFrame) -> DataFrame:
    df = df.dropna(subset=["model_response"])
    df = df.loc[df["model_response"].str.strip() != ""]

    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        try:
            obj = loads(s)
            return obj if isinstance(obj, (dict, list)) and obj else None
        except Exception:
            return None

    df["model_response"] = df["model_response"].map(parse_json)
    df = df.dropna(subset=["model_response"])

    return df


def load_model_responses(db_path: Path) -> DataFrame:
    query = """
SELECT
	impact.doi,
	json_extract(openalex.json_data, '$.publication_year') AS publication_year,
	impact.model_response
FROM
	identify_ptm_impact_analysis impact
INNER JOIN openalex ON impact.doi = openalex.doi;
    """

    with sqlite3.connect(db_path) as conn:
        df: DataFrame = pd.read_sql(query, conn)

    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df.dropna(subset=["publication_year"]).copy()
    df["publication_year"] = df["publication_year"].astype(int)

    return df.sort_values(by=["publication_year", "doi"])


def build_step_frame(df: DataFrame) -> DataFrame:
    df["step"] = df["model_response"].map(extract_steps)
    df = df.explode(column="step")
    df = df.dropna(subset=["step"])
    df = df.loc[df["step"].str.strip() != ""]

    df = df.drop_duplicates(subset=["doi", "step"])
    return df.sort_values(by=["step", "publication_year", "doi"])


def count_unique_dois_per_step(df: DataFrame) -> pd.Series:
    return df.groupby("step")["doi"].nunique().sort_values(ascending=False)


def build_yearly_counts(df: DataFrame) -> DataFrame:
    return (
        df.groupby(["step", "publication_year"], as_index=False)["doi"]
        .nunique()
        .rename(columns={"doi": "count"})
    )


def plot_counts(counts: DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=FIGSIZE,
        sharex=True,
    )
    year_order: list[int] = list(range(2009, 2026))

    for ax, step in zip(axes, SCIENTIFIC_WORKFLOW_STEPS, strict=True):
        step_counts: DataFrame = counts[counts["step"] == step].sort_values(
            by="publication_year",
        )

        sns.barplot(
            data=step_counts,
            x="publication_year",
            y="count",
            order=year_order,
            ax=ax,
        )
        ax.set_title(step, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
        ax.set_ylabel("Papers", fontsize=XY_LABEL_FONT_SIZE)
        ax.set_xticks(range(0, len(year_order), 2))
        ax.set_xticklabels(year_order[::2])
        ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "Papers per Year by Scientific Workflow Stage",
        fontsize=SUPTITLE_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def print_report(df: DataFrame, counts: pd.Series) -> None:
    print(counts.reindex(SCIENTIFIC_WORKFLOW_STEPS, fill_value=0).to_string())
    print()

    for step in SCIENTIFIC_WORKFLOW_STEPS:
        step_df: DataFrame = (
            df[df["step"] == step]
            .sort_values(by=["publication_year", "doi"])
            .head(TOP_N)
        )
        print(step)
        print(step_df[["publication_year", "doi"]].to_string(index=False))
        print()


@click.command()
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    help="Path to the SQLite database.",
)
def main(db_path: Path) -> None:
    db_path = db_path.absolute()

    df: DataFrame = load_model_responses(db_path=db_path)
    df = normalize_data(df=df)

    step_df: DataFrame = build_step_frame(df=df)
    counts: pd.Series = count_unique_dois_per_step(df=step_df)
    yearly_counts: DataFrame = build_yearly_counts(df=step_df)

    plot_counts(counts=yearly_counts, output_path=DEFAULT_OUTPUT_PATH)
    print_report(df=step_df, counts=counts)


if __name__ == "__main__":
    main()
