from json import loads
from pathlib import Path
from textwrap import fill
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
DL_LABEL: str = "PTM Reuse"
NO_DL_LABEL: str = "No PTM Reuse"
FIGSIZE: tuple[float, float] = (12.8, 9.6)


def load_dl_rows(db: Engine) -> DataFrame:
    sql: str = """
SELECT
    udl.doi,
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) AS publication_year,
    udl.model_response
FROM
    uses_ptms_analysis udl
JOIN
    openalex oa
ON
    oa.doi = udl.doi
INNER JOIN
    natural_science_article_dois ns
ON
    ns.doi = udl.doi
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

    fig, ax = plt.subplots(figsize=(10, 8))
    zero_counts = pd.Series(0, index=pivot.index)
    blue_counts = pivot.get(DL_LABEL, zero_counts)
    red_counts = pivot.get(NO_DL_LABEL, zero_counts)

    red_bars = ax.bar(
        pivot["publication_year"],
        red_counts,
        color="#C44E52",
        label=NO_DL_LABEL,
    )
    blue_bars = ax.bar(
        pivot["publication_year"],
        blue_counts,
        color="#4C78A8",
        label=DL_LABEL,
    )

    print(df)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Year", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=XY_LABEL_FONT_SIZE)
    plt.suptitle("Papers Reusing PTMs per Year", fontsize=SUPTITLE_FONT_SIZE)
    plt.title(
        label=f"{df[df['category'] == 'PTM Reuse']['count'].sum():,} Reuse PTMs; {df[df['category'] == 'No PTM Reuse']['count'].sum():,} No PTM Reuse;",
        fontsize=TITLE_FONT_SIZE,
        loc="center",
    )
    xticks: list[int] = years[::2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(year) for year in xticks], rotation=45)
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.legend(title="", fontsize=OTHER_FONT_SIZE)
    ax.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
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
@click.option(
    "--output",
    "output_path",
    default=Path("figW.pdf").absolute(),
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


if __name__ == "__main__":
    main()
