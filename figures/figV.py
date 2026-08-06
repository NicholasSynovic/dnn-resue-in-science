from json import JSONDecodeError, loads
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
from sqlalchemy import Engine, create_engine

SUPTITLE_FONT_SIZE: int = 24
TITLE_FONT_SIZE: int = 22
XY_LABEL_FONT_SIZE: int = 20
XY_TICK_FONT_SIZE: int = 18
OTHER_FONT_SIZE: int = XY_TICK_FONT_SIZE
DL_LABEL: str = "DL Usage"
NO_DL_LABEL: str = "No DL Usage"
FIGSIZE: tuple[float, float] = (12.8, 6.4)


def load_dl_rows(db: Engine) -> DataFrame:
    sql: str = """
SELECT
    udl.doi,
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) AS publication_year,
    udl.model_response
FROM
    uses_dl_analysis udl
JOIN
    openalex oa
ON
    oa.doi = udl.doi
INNER JOIN
    natural_science_article_dois ns
ON
    ns.doi = udl.doi
WHERE
    publication_year < 2026;
"""
    return pd.read_sql(sql=sql, con=db)


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


def create_paper_usage_counts(df: DataFrame) -> DataFrame:
    parsed: DataFrame = df.copy()
    parsed = parsed.assign(
        parsed_model_response=parsed["model_response"].map(parse_json)
    )

    def classify_paper(obj: Any) -> str | None:
        if isinstance(obj, dict) and obj.get("result") is True:
            return DL_LABEL

        if isinstance(obj, dict) and obj.get("result") is False:
            return NO_DL_LABEL

        return None

    parsed = parsed.assign(category=parsed["parsed_model_response"].map(classify_paper))
    parsed = parsed.dropna(subset=["publication_year", "category"])
    parsed = parsed.drop_duplicates(subset=["doi"], keep="first")
    # parsed = parsed[parsed["publication_year"] >= 2012]

    counts: DataFrame = (
        parsed.groupby(["publication_year", "category"])
        .size()
        .reset_index(name="count")
        .sort_values(by="publication_year")
    )

    counts = counts.assign(publication_year=counts["publication_year"].astype(int))

    return counts


def get_dl_dois(df: DataFrame) -> DataFrame:
    parsed: DataFrame = df.copy()
    parsed = parsed.assign(
        parsed_model_response=parsed["model_response"].map(parse_json)
    )

    def is_dl_paper(obj: Any) -> bool:
        return isinstance(obj, dict) and obj.get("result") is True

    parsed = parsed.loc[parsed["parsed_model_response"].map(is_dl_paper)]
    parsed = parsed.dropna(subset=["publication_year"])
    parsed = parsed.drop_duplicates(subset=["doi"], keep="first")
    parsed = parsed.sort_values(by=["publication_year", "doi"])

    return parsed.loc[:, ["publication_year", "doi"]]


def print_dl_dois(df: DataFrame) -> None:
    for _, row in df.iterrows():
        print(row["doi"], row["publication_year"])


def plot_counts(df: DataFrame, output_path: Path) -> None:
    if df.empty:
        raise ValueError("No classified papers available for plotting.")

    min_year: int = int(df["publication_year"].min())
    max_year: int = int(df["publication_year"].max())
    years: list[int] = list(range(min_year, max_year + 1))
    pivot: DataFrame = (
        df.pivot_table(
            index="publication_year",
            columns="category",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reindex(years, fill_value=0)
        .reset_index()
    )

    pivot = pivot.loc[pivot.get(DL_LABEL, pd.Series(0, index=pivot.index)) > 0]
    if pivot.empty:
        raise ValueError("No years with DL papers available for plotting.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    zero_counts = pd.Series(0, index=pivot.index)
    blue_counts = pivot.get(DL_LABEL, zero_counts)
    red_counts = pivot.get(NO_DL_LABEL, zero_counts)

    # True stacking: bar height is the real per-year total, not max(DL, No-DL).
    ax.bar(
        pivot["publication_year"],
        blue_counts,
        color="#4C78A8",
        label=DL_LABEL,
    )
    ax.bar(
        pivot["publication_year"],
        red_counts,
        bottom=blue_counts,
        color="#C44E52",
        label=NO_DL_LABEL,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=XY_LABEL_FONT_SIZE)

    # Stacked above the axes, top to bottom: title, subtitle, legend.
    ax.set_title(
        "Papers Using Deep Learning per Year",
        fontsize=SUPTITLE_FONT_SIZE,
        pad=76,
    )
    dl_total: int = int(df.loc[df["category"] == DL_LABEL, "count"].sum())
    no_dl_total: int = int(df.loc[df["category"] == NO_DL_LABEL, "count"].sum())
    ax.text(
        0.5,
        1.145,
        f"Candidate Papers: {dl_total + no_dl_total:,} | Use DL: {dl_total:,}",
        transform=ax.transAxes,
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="bottom",
    )

    xticks: list[int] = years[::2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(year) for year in xticks], rotation=45)
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        fontsize=OTHER_FONT_SIZE,
    )
    ax.grid(visible=False)
    ax.set_axisbelow(True)

    fig.tight_layout()
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
    default=Path("figV.pdf").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output path for the stacked paper counts plot.",
)
def main(db_path: Path, output_path: Path) -> None:
    db_path = db_path.absolute()
    output_path = output_path.absolute()

    db: Engine = create_engine(url=f"sqlite:///{db_path}")

    dl_df: DataFrame = load_dl_rows(db=db)

    counts: DataFrame = create_paper_usage_counts(df=dl_df)

    plot_counts(df=counts, output_path=output_path)

    print(counts.groupby("category")["count"].sum().to_string())


if __name__ == "__main__":
    main()
