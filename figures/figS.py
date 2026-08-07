from json import JSONDecodeError, loads
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
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
DL_LABEL: str = "PTM Reuse"
NO_DL_LABEL: str = "No PTM Reuse"
FIGSIZE: tuple[float, float] = (12.8, 8.4)

FIRST_YEAR: int = 2016
LAST_YEAR: int = 2025

# Figure-fraction height reserved below the axes for the legend, and the
# height at which the legend's top edge is anchored. The anchor sits above the
# reserved strip so the legend tucks into tight_layout's padding, keeping it
# close to the x-axis label.
LEGEND_RECT_BOTTOM: float = 0.20
LEGEND_TOP: float = 0.225

# Okabe-Ito colourblind-safe palette, assigned in descending-total order. The
# weak yellow sits mid-rank rather than on one of the low-lying lines.
FIELD_COLORS: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#8C6D31",
    "#000000",
]


def load_papers(db: Engine) -> DataFrame:
    sql: str = """
SELECT
    uptm.doi,
    uptm.model_response,
    oa.topic_0,
    oa.topic_1,
    oa.topic_2,
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) AS publication_year
FROM
    uses_ptms_analysis uptm
JOIN
    openalex oa
ON
    oa.doi = uptm.doi
INNER JOIN
    natural_science_article_dois ns
ON
    ns.doi = uptm.doi;
"""
    df: DataFrame = pd.read_sql(sql=sql, con=db)

    return df[df["publication_year"] < 2026]


def parse_json(value: str) -> dict[str, Any] | list[Any] | None:
    if not isinstance(value, str) or value.strip() == "":
        return None

    try:
        parsed: Any = loads(value)
    except (JSONDecodeError, TypeError, ValueError):
        return None

    if isinstance(parsed, dict | list):
        return parsed

    return None


def create_field_dataframes(df: DataFrame) -> dict[str, DataFrame]:
    data: dict[str, list[str | int]] = {
        "year": [],
        "field": [],
        "ptm_reusing": [],
        "no_ptm": [],
    }
    count = 0

    for _, row in df.sort_values(by="publication_year").iterrows():
        parsed_response = parse_json(str(row["model_response"]))
        if not isinstance(parsed_response, dict):
            continue

        result = parsed_response.get("result")
        if result is True:
            ptm_reusing = 1
            no_ptm = 0
            count += 1
        elif result is False:
            ptm_reusing = 0
            no_ptm = 1
        else:
            continue

        year: int = int(row["publication_year"])
        if year < FIRST_YEAR:
            continue

        # A set, so a paper repeating the same topic across topic_0/1/2 counts
        # once for that field. Papers spanning different fields still count in
        # each of them.
        topics: set[str] = {
            str(row["topic_0"]),
            str(row["topic_1"]),
            str(row["topic_2"]),
        }

        for topic in topics:
            if topic in FIELD:
                data["year"].append(year)
                data["field"].append(topic)
                data["ptm_reusing"].append(ptm_reusing)
                data["no_ptm"].append(no_ptm)

    data_df = DataFrame(data=data)
    if data_df.empty:
        return {
            field: DataFrame(columns=["year", "ptm_reusing", "no_ptm"])
            for field in FIELD
        }

    min_year = int(data_df["year"].min())
    max_year = int(data_df["year"].max())
    years = list(range(min_year, max_year + 1))

    field_dataframes: dict[str, DataFrame] = {}
    for field in FIELD:
        field_df = data_df.loc[data_df["field"] == field]
        field_counts = (
            field_df.groupby("year", as_index=False)[["ptm_reusing", "no_ptm"]].sum()
            if not field_df.empty
            else DataFrame(columns=["year", "ptm_reusing", "no_ptm"])
        )

        field_counts = (
            field_counts.set_index("year")
            .reindex(years, fill_value=0)
            .reset_index()
            .rename(columns={"index": "year"})
        )
        field_counts[["ptm_reusing", "no_ptm"]] = field_counts[
            ["ptm_reusing", "no_ptm"]
        ].astype(int)
        field_dataframes[field] = field_counts

    print("PTM reusing papers", count)
    return field_dataframes


def count_papers(df: DataFrame) -> tuple[int, int]:
    uses_dl: int = 0
    no_dl: int = 0

    for _, row in df.iterrows():
        parsed_response = parse_json(str(row["model_response"]))
        if not isinstance(parsed_response, dict):
            continue

        result = parsed_response.get("result")
        if result is True:
            uses_dl += 1
        elif result is False:
            no_dl += 1

    return (uses_dl, no_dl)


def plot(
    field_dataframes: dict[str, DataFrame],
    output_path: Path,
) -> None:
    # One line per field, ordered by descending total so the legend order
    # matches how the lines stack at the right-hand edge.
    totals: dict[str, int] = {
        field: int(df["ptm_reusing"].sum()) for field, df in field_dataframes.items()
    }
    ordered_fields: list[str] = [
        field
        for field, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for field, color in zip(ordered_fields, FIELD_COLORS, strict=True):
        field_df: DataFrame = field_dataframes[field]
        ax.plot(
            field_df["year"],
            field_df["ptm_reusing"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=field,
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(FIRST_YEAR - 0.4, LAST_YEAR + 0.4)
    ax.set_ylim(bottom=0)
    xticks: list[int] = list(range(FIRST_YEAR, LAST_YEAR + 1, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(year) for year in xticks])
    ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Papers Reusing PTMs", fontsize=XY_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(visible=False)
    ax.set_axisbelow(True)

    ax.set_title(
        "Pre-Trained Model (PTM) Reuse per Year Across Scientific Fields",
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
        ncol=2,
        frameon=False,
        fontsize=OTHER_FONT_SIZE,
    )

    fig.tight_layout(rect=(0, LEGEND_RECT_BOTTOM, 1, 1))
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
    default=Path("figS.pdf").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output path for the plot.",
)
def main(db_path: Path, output_path: Path) -> None:
    db_path = db_path.absolute()
    output_path = output_path.absolute()
    db: Engine = create_engine(url=f"sqlite:///{db_path}")

    papers: DataFrame = load_papers(db=db)
    field_dataframes = create_field_dataframes(df=papers)

    uses_dl_count, no_dl_count = count_papers(df=papers)
    for field, field_df in field_dataframes.items():
        reusing: int = field_df["ptm_reusing"].sum()
        print(field, reusing, (reusing / uses_dl_count) * 100)

    print(f"{uses_dl_count:,} reuse PTMs; {no_dl_count:,} do not")

    plot(
        field_dataframes=field_dataframes,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
