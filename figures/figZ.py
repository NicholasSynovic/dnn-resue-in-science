import sqlite3
from json import loads
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame

SUPTITLE_FONT_SIZE: int = 24
TITLE_FONT_SIZE: int = 22
XY_LABEL_FONT_SIZE: int = 20
XY_TICK_FONT_SIZE: int = 18
OTHER_FONT_SIZE: int = XY_TICK_FONT_SIZE
FIGSIZE: tuple[float, float] = (12.8, 9.6)


def plot(df: DataFrame, output_path: Path) -> None:
    df = df.loc[df["publication_year"] >= 2012].copy()

    classification_labels = {
        "Observation": "Observation",
        "Hypothesis": "Hypothesis",
        "Background": "Background",
        "Analysis": "Analysis",
        "Test": "Test",
    }
    classifications = [
        "Test",
        "Analysis",
        "Background",
        "Hypothesis",
        "Observation",
    ]

    counts = (
        df.groupby(["publication_year", "classification"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["classification"] = counts["classification"].replace(classification_labels)
    counts = counts.loc[counts["classification"].isin(classifications)].copy()

    if counts.empty:
        click.echo("No valid classification rows found; skipping plot.")
        return

    year_min = 2012
    year_max = int(counts["publication_year"].max())
    years = list(range(year_min, year_max + 1))

    pivot = (
        counts.pivot_table(
            index="publication_year",
            columns="classification",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(years, fill_value=0)
        .reset_index()
    )

    for classification in classifications:
        if classification not in pivot.columns:
            pivot[classification] = 0

    pivot = pivot[["publication_year"] + classifications]
    pivot[classifications] = pivot[classifications].astype(int)

    totals = {
        classification: int(pivot[classification].sum())
        for classification in classifications
    }
    summary_order = sorted(
        classifications, key=lambda classification: totals[classification], reverse=True
    )
    summary_text = "; ".join(
        f"{totals[classification]:,} {classification}"
        for classification in summary_order
    )
    ymax = max(int(pivot[classifications].to_numpy().max() * 1.15), 1)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    colors = {
        "Observation": "#4C78A8",
        "Hypothesis": "#54A24B",
        "Background": "#C44E52",
        "Analysis": "#F58518",
        "Test": "#8C6D31",
    }

    ax.bar(
        pivot["publication_year"],
        pivot["Test"],
        color=colors["Test"],
        label="Test",
    )
    ax.bar(
        pivot["publication_year"],
        pivot["Analysis"],
        color=colors["Analysis"],
        label="Analysis",
    )
    ax.bar(
        pivot["publication_year"],
        pivot["Background"],
        color=colors["Background"],
        label="Background",
    )
    ax.bar(
        pivot["publication_year"],
        pivot["Hypothesis"],
        color=colors["Hypothesis"],
        label="Hypothesis",
    )
    ax.bar(
        pivot["publication_year"],
        pivot["Observation"],
        color=colors["Observation"],
        label="Observation",
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=XY_LABEL_FONT_SIZE)
    fig.text(
        0.5,
        0.975,
        "PTM Reuse Impact Counts per Year",
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.945,
        summary_text,
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="top",
    )
    ax.set_ylim(0, max(ymax, 1))
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.tick_params(axis="x", rotation=45)

    xtick_labels = [
        str(year) if position % 2 == 0 else "" for position, year in enumerate(years)
    ]
    ax.set_xticks(years)
    ax.set_xticklabels(xtick_labels)

    ax.legend(
        handles=[
            Patch(color=colors[classification], label=classification)
            for classification in classifications
        ],
        frameon=True,
        fontsize=OTHER_FONT_SIZE,
        # title="Scientific Process",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path)
    plt.close(fig)


def extract_classification(obj: Any) -> list[str]:
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
    # 1. Drop NULL and empty / whitespace-only strings.
    df = df.dropna(subset=["model_response"])
    df = df.loc[df["model_response"].str.strip() != ""]

    # 2. Parse JSON safely.
    def parse_json(s: str) -> dict[str, Any] | None:
        try:
            obj = loads(s)
            return obj if isinstance(obj, dict) and obj else None
        except Exception:
            return None

    df["model_response"] = df["model_response"].map(parse_json)

    # 3. Drop malformed JSON or empty dicts.
    df = df.dropna(subset=["model_response"])

    return df


def load_model_responses(db_path: Path) -> DataFrame:
    query = f"""
SELECT
	oa.doi,
	json_extract(
		oa.json_data, '$.publication_year'
	) AS publication_year,
	impact.model_response
FROM
	openalex oa
INNER JOIN natural_science_article_dois ns ON oa.doi = ns.doi
JOIN identify_ptm_impact_analysis impact ON oa.doi = impact.doi;
    """

    with sqlite3.connect(db_path) as conn:
        df: DataFrame = pd.read_sql(query, conn)

    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df.dropna(subset=["publication_year"]).copy()
    df["publication_year"] = df["publication_year"].astype(int)

    return df[df["publication_year"] < 2026].sort_values(by="publication_year")


@click.command()
@click.option(
    "--db",
    "db_path",
    default=Path("../data/aius_12-17-2025.db").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to the SQLite database.",
)
@click.option(
    "--output",
    "output_path",
    default=Path("figZ.pdf").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output path for the plot.",
)
def main(db_path: Path, output_path: Path) -> None:
    db_path = db_path.absolute()
    output_path = output_path.absolute()

    df: DataFrame = load_model_responses(db_path=db_path)
    df = normalize_data(df=df)

    df["classification"] = df["model_response"].map(extract_classification)
    df = df.explode(column="classification")
    df = df.dropna(subset=["classification"]).copy()
    df = df.loc[df["classification"].map(lambda value: isinstance(value, str))].copy()

    plot(df, output_path=output_path)


if __name__ == "__main__":
    main()
