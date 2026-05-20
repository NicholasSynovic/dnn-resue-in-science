from json import loads
from pathlib import Path
from textwrap import fill
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import BarContainer
from matplotlib.patches import Patch
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
DL_LABEL: str = "DL Usage"
NO_DL_LABEL: str = "No DL Usage"
FIGSIZE: tuple[float, float] = (24, 10)


def load_papers(db: Engine) -> DataFrame:
    sql: str = """
SELECT
    udl.doi,
    udl.model_response,
    oa.topic_0,
    oa.topic_1,
    oa.topic_2,
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) AS publication_year
FROM
    uses_dl_analysis udl
JOIN
    openalex oa
ON
    oa.doi = udl.doi;
"""
    df: DataFrame = pd.read_sql(sql=sql, con=db)

    return df[df["publication_year"] < 2026]


def parse_json(value: str) -> dict[str, Any] | list[Any] | None:
    if not isinstance(value, str) or value.strip() == "":
        return None

    try:
        parsed: Any = loads(value)
    except Exception:
        return None

    if isinstance(parsed, dict | list):
        return parsed

    return None


def create_field_dataframes(df: DataFrame) -> dict[str, DataFrame]:
    count: int = 0
    data: dict[str, list[str | int]] = {
        "year": [],
        "field": [],
        "dl_using": [],
        "no_dl": [],
    }

    for _, row in df.iterrows():
        parsed_response = parse_json(str(row["model_response"]))
        if not isinstance(parsed_response, dict):
            continue

        result = parsed_response.get("result")
        if result is True:
            count += 1
            dl_using = 1
            no_dl = 0
        elif result is False:
            dl_using = 0
            no_dl = 1
        else:
            continue

        topics: list[str] = [
            str(row["topic_0"]),
            str(row["topic_1"]),
            str(row["topic_2"]),
        ]

        for topic in topics:
            if int(row["publication_year"]) < 2012:
                continue

            if topic in FIELD:
                data["year"].append(int(row["publication_year"]))
                data["field"].append(topic)
                data["dl_using"].append(dl_using)
                data["no_dl"].append(no_dl)

    data_df = DataFrame(data=data)
    if data_df.empty:
        return {
            field: DataFrame(columns=["year", "dl_using", "no_dl"]) for field in FIELD
        }

    min_year = int(data_df["year"].min())
    max_year = int(data_df["year"].max())
    years = list(range(min_year, max_year + 1))

    field_dataframes: dict[str, DataFrame] = {}
    for field in FIELD:
        field_df = data_df.loc[data_df["field"] == field]
        field_counts = (
            field_df.groupby("year", as_index=False)[["dl_using", "no_dl"]].sum()
            if not field_df.empty
            else DataFrame(columns=["year", "dl_using", "no_dl"])
        )

        field_counts = (
            field_counts.set_index("year")
            .reindex(years, fill_value=0)
            .reset_index()
            .rename(columns={"index": "year"})
        )
        field_counts[["dl_using", "no_dl"]] = field_counts[
            ["dl_using", "no_dl"]
        ].astype(int)
        field_dataframes[field] = field_counts

    print("DL using papers", count)
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
        else:
            no_dl += 1

    return (uses_dl, no_dl)


def plot(
    field_dataframes: dict[str, DataFrame],
    uses_dl_count: int,
    no_dl_count: int,
    output_path: Path,
) -> None:
    panel_labels: list[str] = [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(H)",
    ]

    totals: dict[str, int] = {
        field: int(df["dl_using"].sum()) for field, df in field_dataframes.items()
    }

    ordered_fields = [
        field
        for field, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=FIGSIZE, sharey="row")
    flat_axes = axes.flatten()
    top_row_fields = ordered_fields[:4]
    bottom_row_fields = ordered_fields[4:]
    top_row_max = max(
        (
            field_dataframes[field][["dl_using", "no_dl"]].sum(axis=1).max()
            for field in top_row_fields
        ),
        default=0,
    )
    bottom_row_max = max(
        (
            field_dataframes[field][["dl_using", "no_dl"]].sum(axis=1).max()
            for field in bottom_row_fields
        ),
        default=0,
    )
    row_max = [top_row_max, bottom_row_max]

    for index, field in enumerate(ordered_fields):
        ax = flat_axes[index]
        panel_data: DataFrame = field_dataframes[field]
        wrapped_title = fill(field, width=30) if len(field) > 30 else field

        blue_bars = ax.bar(
            panel_data["year"],
            panel_data["dl_using"],
            color="#4C78A8",
            label=DL_LABEL,
        )
        red_bars = ax.bar(
            panel_data["year"],
            panel_data["no_dl"],
            # bottom=panel_data["dl_using"],
            color="#C44E52",
            label=NO_DL_LABEL,
        )

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        row_index = index // 4
        ax.set_ylim(0, row_max[row_index] * 1.3 if row_max[row_index] else 1)
        ax.set_title(wrapped_title, fontsize=TITLE_FONT_SIZE)

        if row_index == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)

        years: list[int] = list(range(2012, 2026, 2))
        ax.set_xticks(years)
        ax.set_xticklabels([str(year) for year in years])

        if index == 0 or index == 4:
            ax.set_ylabel("Paper Count", fontsize=XY_LABEL_FONT_SIZE)

        ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45)

        # ax.bar_label(blue_bars, fmt="{:,.0f}", padding=3, fontsize=OTHER_FONT_SIZE)
        # ax.bar_label(red_bars, fmt="{:,.0f}", padding=3, fontsize=OTHER_FONT_SIZE)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.text(
            0.02,
            0.98,
            panel_labels[index],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )

        if index == 0:
            ax.legend(
                handles=[
                    Patch(color="#4C78A8", label=DL_LABEL),
                    Patch(color="#C44E52", label=NO_DL_LABEL),
                ],
                loc="center left",
                frameon=True,
                fontsize=OTHER_FONT_SIZE,
            )

    fig.suptitle(
        "Number Of Papers Using Deep Learning per Year",
        fontsize=SUPTITLE_FONT_SIZE,
        y=0.99,
    )

    fig.text(
        0.5,
        0.955,
        f"{uses_dl_count + no_dl_count:,} Papers Analyzed; {uses_dl_count:,} Papers Using Deep Learning",
        ha="center",
        va="top",
        fontsize=TITLE_FONT_SIZE,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path)
    plt.close(fig)


@click.command()
@click.option(
    "--db",
    "db_path",
    default=Path("../data/aius_12-17-2025.db").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to the SQLite database.",
)
def main(db_path: Path) -> None:
    db_path = db_path.absolute()
    db: Engine = create_engine(url=f"sqlite:///{db_path}")

    papers: DataFrame = load_papers(db=db)
    field_dataframes = create_field_dataframes(df=papers)

    for field, field_df in field_dataframes.items():
        print(field, field_df["dl_using"].sum())

    uses_dl_count, no_dl_count = count_papers(df=papers)

    plot(
        field_dataframes=field_dataframes,
        uses_dl_count=uses_dl_count,
        no_dl_count=no_dl_count,
        output_path=Path("figR.pdf").absolute(),
    )


if __name__ == "__main__":
    main()
