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
        "adaptation_reuse": "Adaptation Reuse",
        "conceptual_reuse": "Conceptual Reuse",
        "deployment_reuse": "Deployment Reuse",
    }
    classifications = [
        "Adaptation Reuse",
        "Conceptual Reuse",
        "Deployment Reuse",
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
    ymax = 300

    fig, ax = plt.subplots(figsize=FIGSIZE)

    colors = {
        "Adaptation Reuse": "#4C78A8",
        "Conceptual Reuse": "#54A24B",
        "Deployment Reuse": "#C44E52",
    }

    ax.bar(
        pivot["publication_year"],
        pivot["Adaptation Reuse"],
        # bottom=bottom,
        color=colors["Adaptation Reuse"],
        label="Adaptation Reuse",
    )

    ax.bar(
        pivot["publication_year"],
        pivot["Deployment Reuse"],
        # bottom=bottom,
        color=colors["Deployment Reuse"],
        label="Deployment Reuse",
    )

    ax.bar(
        pivot["publication_year"],
        pivot["Conceptual Reuse"],
        # bottom=bottom,
        color=colors["Conceptual Reuse"],
        label="Conceptual Reuse",
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=XY_LABEL_FONT_SIZE)
    fig.text(
        0.5,
        0.975,
        "PTM Reuse Pattern Counts per Year",
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.945,
        f"{totals['Adaptation Reuse']:,} Adaptation Reuse; {totals['Conceptual Reuse']:,} Conceptual Reuse; {totals['Deployment Reuse']:,} Deployment Reuse",
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
        title="",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path)
    plt.close(fig)


def extract_classification(obj: Any) -> str | None:
    if isinstance(obj, dict):
        value = obj.get("classification")
        return value if isinstance(value, str) and value else None

    return None


def normalize_data(df: DataFrame) -> DataFrame:
    total_rows = len(df)

    # 1. Drop NULL and empty / whitespace-only strings.
    df = df.copy()
    null_rows = int(df["model_response"].isna().sum())
    df = df.dropna(subset=["model_response"]).copy()
    empty_rows = int(df["model_response"].astype(str).str.strip().eq("").sum())
    df = df.loc[df["model_response"].astype(str).str.strip() != ""].copy()

    # 2. Parse JSON safely and keep dicts only.
    def parse_json(s: str) -> tuple[dict[str, Any] | None, str]:
        try:
            obj = loads(s)
            if isinstance(obj, dict) and obj:
                return obj, "ok"
            return None, "non_dict"
        except Exception:
            return None, "invalid_json"

    parsed = df["model_response"].map(parse_json)
    df["model_response"] = parsed.map(lambda item: item[0])
    status = parsed.map(lambda item: item[1])

    invalid_json_rows = int((status == "invalid_json").sum())
    non_dict_rows = int((status == "non_dict").sum())
    df = df.loc[status == "ok"].copy()

    kept_rows = len(df)
    dropped_rows = total_rows - kept_rows
    click.echo(
        "model_response summary: "
        f"kept {kept_rows}/{total_rows} rows; "
        f"dropped {dropped_rows} "
        f"(null={null_rows}, empty={empty_rows}, "
        f"invalid_json={invalid_json_rows}, non_dict={non_dict_rows})"
    )

    return df


def load_model_responses(db_path: Path) -> DataFrame:
    query = f"""
        SELECT
  oa.doi,
  json_extract(
    oa.json_data, '$.publication_year'
  ) AS publication_year,
  reuse.model_response
FROM
  openalex oa
  INNER JOIN natural_science_article_dois ns ON oa.doi = ns.doi
  JOIN identify_ptm_reuse_analysis reuse ON oa.doi = reuse.doi;
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
    default=Path("figT.pdf").absolute(),
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

    missing_classification_rows = int(df["classification"].isna().sum())
    if missing_classification_rows:
        click.echo(
            "classification summary: "
            f"dropped {missing_classification_rows} rows with no classification key"
        )

    df = df.dropna(subset=["classification"]).copy()

    df = df.loc[df["publication_year"] >= 2012].copy()

    class_totals = (
        df.loc[
            df["classification"].isin(
                ["adaptation_reuse", "deployment_reuse", "conceptual_reuse"]
            )
        ]
        .groupby("classification")
        .size()
        .to_dict()
    )

    plot(df, output_path=output_path)

    print(
        "reuse totals: ",
        f"adaptation={class_totals.get('adaptation_reuse', 0)}, ",
        f"deployment={class_totals.get('deployment_reuse', 0)}, ",
        f"conceptual={class_totals.get('conceptual_reuse', 0)}",
    )


if __name__ == "__main__":
    main()
