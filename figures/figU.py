import sqlite3
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pandas import DataFrame

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
FIGSIZE: tuple[float, float] = (12.8, 8.4)

FIRST_YEAR: int = 2012
LAST_YEAR: int = 2025

# Y-axis ceilings round up to a multiple of this.
Y_AXIS_STEP: int = 5

# Tick spacing widens with the height of the axis: taller panels would otherwise
# carry too many labels, shorter ones too few. Each interval divides every
# ceiling it applies to, so the topmost tick always lands on the axis limit.
WIDE_TICK_CEILING: int = 60
MID_TICK_CEILING: int = 30
WIDE_TICK_INTERVAL: int = 20
MID_TICK_INTERVAL: int = 10
NARROW_TICK_INTERVAL: int = 5

# Figure-fraction height reserved below the axes for the legend, and the
# height at which the legend's top edge is anchored. The anchor sits above the
# reserved strip so the legend tucks into tight_layout's padding, keeping it
# close to the x-axis label.
LEGEND_RECT_BOTTOM: float = 0.12
LEGEND_TOP: float = 0.115

# Okabe-Ito colourblind-safe palette, assigned in descending-total order so the
# highest-volume stage takes the strongest colour.
STEP_COLORS: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
]


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
        except (JSONDecodeError, TypeError):
            return None
        return obj if isinstance(obj, (dict, list)) and obj else None

    df = df.copy()
    df["model_response"] = df["model_response"].map(parse_json)
    return df.dropna(subset=["model_response"])


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
    df = df.copy()
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


def pivot_counts(counts: DataFrame) -> DataFrame:
    """Reshape long counts into a year-by-stage matrix over the plotted window."""
    # Years absent from the data are filled with zero so that gap years plot as
    # genuine zeros rather than being bridged by line interpolation.
    window: DataFrame = counts[
        counts["publication_year"].between(FIRST_YEAR, LAST_YEAR)
        & counts["step"].isin(SCIENTIFIC_WORKFLOW_STEPS)
    ]
    matrix: DataFrame = window.pivot_table(
        index="publication_year",
        columns="step",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.reindex(
        columns=SCIENTIFIC_WORKFLOW_STEPS,
        fill_value=0,
    )
    return matrix.reindex(range(FIRST_YEAR, LAST_YEAR + 1), fill_value=0)


def tick_interval(ceiling: int) -> int:
    # Step by 20 above 60, by 10 from 40 through 60, and by 5 at or below 30.
    if ceiling > WIDE_TICK_CEILING:
        return WIDE_TICK_INTERVAL

    if ceiling > MID_TICK_CEILING:
        return MID_TICK_INTERVAL

    return NARROW_TICK_INTERVAL


def axis_ceiling(value: int) -> int:
    # Round the plotted maximum up to the next multiple of Y_AXIS_STEP, then up
    # again to a multiple of the tick interval so the topmost tick lands on the
    # limit instead of stopping short. Only the widest interval can force the
    # second step, and widening never drops a ceiling into a lower band, so one
    # pass is enough.
    if value <= 0:
        return Y_AXIS_STEP

    ceiling: int = -(-value // Y_AXIS_STEP) * Y_AXIS_STEP
    interval: int = tick_interval(ceiling)

    return -(-ceiling // interval) * interval


def plot_counts(counts: DataFrame, output_path: Path) -> None:
    matrix: DataFrame = pivot_counts(counts=counts)

    # One line per stage, ordered by descending total so the legend order
    # matches how the lines stack at the right-hand edge.
    totals: dict[str, int] = {step: int(matrix[step].sum()) for step in matrix.columns}
    ordered_steps: list[str] = [
        step for step, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for step, color in zip(ordered_steps, STEP_COLORS, strict=True):
        ax.plot(
            matrix.index,
            matrix[step],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=step,
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(FIRST_YEAR - 0.4, LAST_YEAR + 0.4)
    # An explicit ceiling, rather than autoscale's 5% padding, which would end
    # the axis on an arbitrary value and push the topmost tick off the figure.
    ceiling: int = axis_ceiling(int(matrix.to_numpy().max()))
    ax.set_ylim(0, ceiling)
    ax.yaxis.set_major_locator(MultipleLocator(tick_interval(ceiling)))
    xticks: list[int] = list(range(FIRST_YEAR, LAST_YEAR + 1, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(year) for year in xticks])
    ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Paper Count", fontsize=XY_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(visible=False)
    ax.set_axisbelow(True)

    ax.set_title(
        "Identified PTM Reuse Impacted Scientific Process Step per Year",
        fontsize=SUPTITLE_FONT_SIZE,
        pad=12,
    )

    # A figure-level legend below the axes. Its top edge is anchored just under
    # the reserved strip, so it sits close to the x-axis label instead of being
    # pushed to the bottom of the figure.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_TOP),
        ncol=3,
        frameon=False,
        fontsize=OTHER_FONT_SIZE,
    )

    fig.tight_layout(rect=(0, LEGEND_RECT_BOTTOM, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def print_report(df: DataFrame, counts: pd.Series, yearly_counts: DataFrame) -> None:
    print(counts.reindex(SCIENTIFIC_WORKFLOW_STEPS, fill_value=0).to_string())
    print()

    # The year-by-stage matrix behind the figure, so the plotted values can be
    # checked without opening the PDF.
    print(f"Stage counts per year, {FIRST_YEAR}-{LAST_YEAR}")
    print(pivot_counts(counts=yearly_counts).to_string())
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
    default=Path("../data/aius.3-18-2026.db").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to the SQLite database.",
)
@click.option(
    "--output",
    "output_path",
    default=Path("figU.pdf").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output path for the plot.",
)
def main(db_path: Path, output_path: Path) -> None:
    db_path = db_path.absolute()
    output_path = output_path.absolute()

    df: DataFrame = load_model_responses(db_path=db_path)
    df = normalize_data(df=df)

    step_df: DataFrame = build_step_frame(df=df)
    counts: pd.Series = count_unique_dois_per_step(df=step_df)
    yearly_counts: DataFrame = build_yearly_counts(df=step_df)

    plot_counts(counts=yearly_counts, output_path=output_path)
    print_report(df=step_df, counts=counts, yearly_counts=yearly_counts)


if __name__ == "__main__":
    main()
