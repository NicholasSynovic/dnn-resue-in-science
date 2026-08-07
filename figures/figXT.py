from json import JSONDecodeError, loads
from pathlib import Path
from textwrap import fill
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pandas import DataFrame
from sqlalchemy import Engine, create_engine

FIELD: list[str] = [
    "Biochemistry, Genetics and Molecular Biology",
    "Neuroscience",
    "Environmental Science",
    "Agricultural and Biological Sciences",
    "Chemistry",
    "Earth and Planetary Sciences",
    "Immunology and Microbiology",
    "Physics and Astronomy",
]

SUPTITLE_FONT_SIZE: int = 24
TITLE_FONT_SIZE: int = 22
XY_LABEL_FONT_SIZE: int = 20
XY_TICK_FONT_SIZE: int = 18
OTHER_FONT_SIZE: int = XY_TICK_FONT_SIZE
FIGSIZE: tuple[float, float] = (30, 10)

FIRST_YEAR: int = 2012
LAST_YEAR: int = 2025

ADAPTATION_LABEL: str = "Adaptation Reuse"
DEPLOYMENT_LABEL: str = "Deployment Reuse"
CONCEPTUAL_LABEL: str = "Conceptual Reuse"

# Ordered by descending total, so the legend order matches the visual
# prominence of the lines.
CLASS_ORDER: list[str] = [
    ADAPTATION_LABEL,
    DEPLOYMENT_LABEL,
    CONCEPTUAL_LABEL,
]

CLASSIFICATION_MAP: dict[str, str] = {
    "adaptation_reuse": ADAPTATION_LABEL,
    "deployment_reuse": DEPLOYMENT_LABEL,
    "conceptual_reuse": CONCEPTUAL_LABEL,
}

# Okabe-Ito colourblind-safe palette, matching the treatment used by the other
# reuse figures.
CLASS_COLORS: dict[str, str] = {
    ADAPTATION_LABEL: "#0072B2",
    DEPLOYMENT_LABEL: "#D55E00",
    CONCEPTUAL_LABEL: "#009E73",
}

PANEL_LABELS: list[str] = [
    "(A)",
    "(B)",
    "(C)",
    "(D)",
    "(E)",
    "(F)",
    "(G)",
    "(H)",
    "(I)",
]

# Figure-fraction geometry. The legend anchor sits above the reserved strip so
# it tucks into tight_layout's padding rather than sinking to the page edge.
LEGEND_RECT_BOTTOM: float = 0.09
LEGEND_TOP: float = 0.085
TITLE_RECT_TOP: float = 0.965
SUBTITLE_Y: float = 0.95

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


def load_papers(db: Engine) -> DataFrame:
    sql: str = """
SELECT
    oa.doi,
    oa.topic_0,
    oa.topic_1,
    oa.topic_2,
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) AS publication_year,
    reuse.model_response
FROM
    openalex oa
INNER JOIN
    natural_science_article_dois ns
ON
    oa.doi = ns.doi
JOIN
    identify_ptm_reuse_analysis reuse
ON
    oa.doi = reuse.doi;
"""
    df: DataFrame = pd.read_sql(sql=sql, con=db)

    return df[df["publication_year"] < 2026]


def parse_json(value: str) -> dict[str, Any] | None:
    if not isinstance(value, str) or value.strip() == "":
        return None

    try:
        parsed: Any = loads(value)
    except (JSONDecodeError, TypeError, ValueError):
        return None

    if isinstance(parsed, dict) and parsed:
        return parsed

    return None


def extract_classification(obj: Any) -> str | None:
    if isinstance(obj, dict):
        value = obj.get("classification")
        return value if isinstance(value, str) and value else None

    return None


def empty_counts_frame(years: list[int]) -> DataFrame:
    frame = DataFrame({"year": years})
    for label in CLASS_ORDER:
        frame[label] = 0

    return frame


def pivot_counts(records: DataFrame, years: list[int]) -> DataFrame:
    # Long -> wide, reindexed onto the full year range so every panel shares an
    # x-axis even where a classification never occurs.
    if records.empty:
        return empty_counts_frame(years=years)

    counts = (
        records.groupby(["year", "classification"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    pivot = counts.pivot_table(
        index="year",
        columns="classification",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.reindex(years, fill_value=0).reset_index()
    pivot = pivot.rename(columns={"index": "year"})

    for label in CLASS_ORDER:
        if label not in pivot.columns:
            pivot[label] = 0

    pivot = pivot[["year", *CLASS_ORDER]].fillna(0)
    pivot[CLASS_ORDER] = pivot[CLASS_ORDER].astype(int)

    return pivot


def create_dataframes(df: DataFrame) -> tuple[DataFrame, dict[str, DataFrame]]:
    years: list[int] = list(range(FIRST_YEAR, LAST_YEAR + 1))

    # Two passes over the same rows. The aggregate counts each paper once; the
    # per-field records explode a paper across every field it belongs to, so
    # the field panels intentionally sum above the aggregate.
    aggregate_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    skipped_no_classification: int = 0
    skipped_unknown: int = 0
    skipped_early: int = 0

    for _, row in df.iterrows():
        parsed_response = parse_json(str(row["model_response"]))
        classification = extract_classification(parsed_response)
        if classification is None:
            skipped_no_classification += 1
            continue

        if classification not in CLASSIFICATION_MAP:
            skipped_unknown += 1
            continue

        year: int = int(row["publication_year"])
        if year < FIRST_YEAR:
            skipped_early += 1
            continue

        label: str = CLASSIFICATION_MAP[classification]
        aggregate_rows.append({"year": year, "classification": label})

        # A set, so a paper repeating the same topic across topic_0/1/2 counts
        # once for that field. Papers spanning different fields still count in
        # each of them.
        topics: set[str] = {
            str(row["topic_0"]),
            str(row["topic_1"]),
            str(row["topic_2"]),
        }

        field_rows.extend(
            {"year": year, "field": topic, "classification": label}
            for topic in topics
            if topic in FIELD
        )

    click.echo(
        "row summary: "
        f"kept {len(aggregate_rows)}; "
        f"dropped {skipped_no_classification} without a classification key, "
        f"{skipped_unknown} outside the reuse patterns, "
        f"{skipped_early} before {FIRST_YEAR}"
    )

    aggregate_df = pivot_counts(records=DataFrame(aggregate_rows), years=years)

    field_records = DataFrame(field_rows)
    field_dataframes: dict[str, DataFrame] = {}
    for field in FIELD:
        subset = (
            field_records.loc[field_records["field"] == field]
            if not field_records.empty
            else field_records
        )
        field_dataframes[field] = pivot_counts(records=subset, years=years)

    return aggregate_df, field_dataframes


def panel_max(panel_data: DataFrame) -> int:
    # Each classification is its own line, so the ceiling is the largest single
    # value rather than the per-year total.
    if panel_data.empty:
        return 0

    return int(panel_data[CLASS_ORDER].to_numpy().max())


def tick_interval(ceiling: int) -> int:
    # Step by 20 above 60, by 10 from 40 through 60, and by 5 at or below 30.
    if ceiling > WIDE_TICK_CEILING:
        return WIDE_TICK_INTERVAL

    if ceiling > MID_TICK_CEILING:
        return MID_TICK_INTERVAL

    return NARROW_TICK_INTERVAL


def axis_ceiling(value: int) -> int:
    # Round a panel maximum up to the next multiple of Y_AXIS_STEP, then up
    # again to a multiple of the tick interval so the topmost tick lands on the
    # limit instead of stopping short. Only the widest interval can force the
    # second step, and widening never drops a ceiling into a lower band, so one
    # pass is enough.
    if value <= 0:
        return Y_AXIS_STEP

    ceiling: int = -(-value // Y_AXIS_STEP) * Y_AXIS_STEP
    interval: int = tick_interval(ceiling)

    return -(-ceiling // interval) * interval


def draw_panel(
    ax: Axes,
    panel_data: DataFrame,
    title: str,
    panel_label: str,
    ymax: int,
) -> None:
    # One line per reuse pattern, coloured consistently across every panel so a
    # given colour means the same classification throughout the figure.
    for label in CLASS_ORDER:
        ax.plot(
            panel_data["year"],
            panel_data[label],
            marker="o",
            markersize=4,
            linewidth=2,
            color=CLASS_COLORS[label],
            label=label,
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(FIRST_YEAR - 0.4, LAST_YEAR + 0.4)
    xticks: list[int] = list(range(FIRST_YEAR, LAST_YEAR + 1, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(year) for year in xticks])
    # Each panel scales independently, with the ceiling rounded up to a
    # multiple of Y_AXIS_STEP and the tick interval chosen to suit it.
    ceiling: int = axis_ceiling(ymax)
    ax.set_ylim(0, ceiling)
    ax.yaxis.set_major_locator(MultipleLocator(tick_interval(ceiling)))
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(visible=False)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.text(
        0.02,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )


def draw_field_panels(
    fig: Figure,
    grid: GridSpec,
    field_dataframes: dict[str, DataFrame],
    ordered_fields: list[str],
) -> None:
    # Sorting by total puts the four large fields in the top row and the four
    # small ones in the bottom row. Every panel scales independently, so each
    # one carries its own tick labels.
    row_axes: list[list[Axes]] = []
    for row in range(2):
        axes_in_row: list[Axes] = [
            fig.add_subplot(grid[row, column]) for column in range(1, 5)
        ]
        row_axes.append(axes_in_row)

    for index, field in enumerate(ordered_fields):
        row = index // 4
        column = index % 4
        ax = row_axes[row][column]
        wrapped_title = fill(field, width=26) if len(field) > 26 else field
        draw_panel(
            ax=ax,
            panel_data=field_dataframes[field],
            title=wrapped_title,
            panel_label=PANEL_LABELS[index + 1],
            ymax=panel_max(panel_data=field_dataframes[field]),
        )

        if row == 1:
            ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)

        if column == 0:
            ax.set_ylabel("Paper Count", fontsize=XY_LABEL_FONT_SIZE)


def plot(
    aggregate_df: DataFrame,
    field_dataframes: dict[str, DataFrame],
    output_path: Path,
) -> None:
    totals: dict[str, int] = {
        field: int(df[CLASS_ORDER].to_numpy().sum())
        for field, df in field_dataframes.items()
    }
    ordered_fields: list[str] = [
        field
        for field, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]

    fig: Figure = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(nrows=2, ncols=5)

    # The aggregate spans both rows, giving it height proportional to its
    # larger values; the eight field panels keep a familiar 2x4 block.
    aggregate_ax: Axes = fig.add_subplot(grid[:, 0])
    draw_panel(
        ax=aggregate_ax,
        panel_data=aggregate_df,
        title="Overall",
        panel_label=PANEL_LABELS[0],
        ymax=panel_max(panel_data=aggregate_df),
    )
    aggregate_ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    aggregate_ax.set_ylabel("Paper Count", fontsize=XY_LABEL_FONT_SIZE)

    draw_field_panels(
        fig=fig,
        grid=grid,
        field_dataframes=field_dataframes,
        ordered_fields=ordered_fields,
    )

    fig.suptitle(
        "Pre-Trained Model Reuse Patterns per Year",
        fontsize=SUPTITLE_FONT_SIZE,
        y=0.99,
    )

    fig.text(
        0.5,
        SUBTITLE_Y,
        "Overall and by Field",
        ha="center",
        va="top",
        fontsize=OTHER_FONT_SIZE,
    )

    handles, labels = aggregate_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_TOP),
        ncol=3,
        frameon=False,
        fontsize=OTHER_FONT_SIZE,
    )

    fig.tight_layout(rect=(0, LEGEND_RECT_BOTTOM, 1, TITLE_RECT_TOP))
    fig.savefig(output_path)
    plt.close(fig)


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
    default=Path("figXT.pdf").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output path for the plot.",
)
def main(db_path: Path, output_path: Path) -> None:
    db_path = db_path.absolute()
    output_path = output_path.absolute()
    db: Engine = create_engine(url=f"sqlite:///{db_path}")

    papers: DataFrame = load_papers(db=db)
    aggregate_df, field_dataframes = create_dataframes(df=papers)

    for label in CLASS_ORDER:
        print(label, int(aggregate_df[label].sum()))

    for field, field_df in field_dataframes.items():
        print(field, int(field_df[CLASS_ORDER].to_numpy().sum()))

    plot(
        aggregate_df=aggregate_df,
        field_dataframes=field_dataframes,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
