from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame, Series
from sqlalchemy import Engine, create_engine

SUPTITLE_FONT_SIZE: int = 24
TITLE_FONT_SIZE: int = 22
XY_LABEL_FONT_SIZE: int = 20
XY_TICK_FONT_SIZE: int = 18
OTHER_FONT_SIZE: int = 18
BAR_LABEL_FONT_SIZE: int = 11
FIGSIZE: tuple[float, float] = (12.8, 8.4)

# Megajournals ordered by descending total paper count. The keys are the values
# stored in `articles.megajournal`; the labels are what gets drawn on the axis.
JOURNAL_KEYS: list[str] = ["PLOS", "FrontiersIn", "BMJ", "F1000"]
JOURNAL_LABELS: list[str] = ["PLOS", "Frontiers", "BMJ", "F1000"]

# Internal (wide-DataFrame) column names, in funnel order.
CATEGORIES: list[str] = [
    "Total Papers",
    "OpenAlex Indexed Papers",
    "Papers With Citations",
    "Natural Science Papers",
    "JATS XML Documents",
]

# Abbreviated names used in the legend.
CATEGORY_LABELS: dict[str, str] = {
    "Total Papers": "Total",
    "OpenAlex Indexed Papers": "OpenAlex",
    "Papers With Citations": "Cited",
    "Natural Science Papers": "Natural science",
    "JATS XML Documents": "JATS XML",
}

CATEGORY_COLORS: dict[str, str] = {
    "Total Papers": "#4C78A8",
    "OpenAlex Indexed Papers": "#F58518",
    "Papers With Citations": "#54A24B",
    "Natural Science Papers": "#C44E52",
    "JATS XML Documents": "#8C6D31",
}

BAR_HEIGHT: float = 0.16
GROUP_PITCH: float = 1.0


def get_papers_per_journal(db: Engine) -> DataFrame:
    all_papers_sql: str = """
SELECT DISTINCT
    doi,
    megajournal
FROM
    articles;
"""
    all_papers_df: DataFrame = pd.read_sql_query(sql=all_papers_sql, con=db)

    doi_year_sql: str = """
SELECT DISTINCT
    doi,
    CAST(json_extract(json_data, '$.publication_year') AS INTEGER)
        AS publication_year
FROM
    openalex;
"""
    doi_year_df: DataFrame = pd.read_sql_query(sql=doi_year_sql, con=db)

    merged_df: DataFrame = pd.merge(all_papers_df, doi_year_df, on="doi", how="left")
    merged_df = merged_df.assign(
        publication_year=merged_df["publication_year"].fillna(0).astype(int)
    )

    return merged_df[merged_df["publication_year"] < 2026]


def get_openalex_papers_per_journal(db: Engine) -> DataFrame:
    sql: str = """
SELECT DISTINCT
    a.megajournal,
    oa.doi
FROM
    articles a
JOIN
    openalex oa ON oa.doi = a.doi
WHERE
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) < 2026;
"""
    return pd.read_sql(sql=sql, con=db)


def get_papers_with_citations_per_journal(db: Engine) -> DataFrame:
    sql: str = """
SELECT DISTINCT
    a.megajournal,
    oa.doi
FROM
    articles a
JOIN
    openalex oa
ON
    oa.doi == a.doi
WHERE
    oa.cited_by_count > 0 AND
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) < 2026;
"""
    return pd.read_sql(sql=sql, con=db)


def get_natural_science_papers_per_journal(db: Engine) -> DataFrame:
    sql: str = """
SELECT DISTINCT
    ns.doi,
    a.megajournal
FROM
    articles a
JOIN
    natural_science_article_dois ns
ON
    ns.doi == a.doi
JOIN
    openalex oa
ON
    oa.doi = ns.doi
WHERE
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) < 2026;
"""
    return pd.read_sql(sql=sql, con=db)


def get_jats_per_journal(db: Engine) -> DataFrame:
    sql: str = """
SELECT DISTINCT
    j.doi,
    a.megajournal
FROM
    articles a
JOIN
    openalex oa
ON
    oa.doi = a.doi
JOIN
    jats j
ON
    a.doi = j.doi
WHERE
    CAST(json_extract(oa.json_data, '$.publication_year') AS INTEGER) < 2026;
"""
    return pd.read_sql(sql=sql, con=db)


def create_data(
    df1: DataFrame, df2: DataFrame, df3: DataFrame, df4: DataFrame, df5: DataFrame
) -> DataFrame:
    # Journals absent from a category count as 0 rather than raising.
    frames: dict[str, DataFrame] = dict(
        zip(CATEGORIES, [df1, df2, df3, df4, df5], strict=True)
    )

    def _counts(df: DataFrame) -> Series:
        return (
            df["megajournal"]
            .value_counts()
            .reindex(JOURNAL_KEYS, fill_value=0)
            .astype(int)
        )

    data: dict[str, list[str] | Series] = {"journal": JOURNAL_KEYS}
    data.update({key: _counts(df).to_list() for key, df in frames.items()})

    return DataFrame(data=data)


def plot(df: DataFrame, output_path: Path) -> None:
    # Draw a horizontal grouped bar chart of paper counts per megajournal.
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Centre each journal's group of five bars on its tick.
    centers = np.arange(len(JOURNAL_KEYS)) * GROUP_PITCH
    offsets = (np.arange(len(CATEGORIES)) - (len(CATEGORIES) - 1) / 2) * BAR_HEIGHT

    for category, offset in zip(CATEGORIES, offsets, strict=True):
        ax.barh(
            centers + offset,
            df[category],
            height=BAR_HEIGHT,
            color=CATEGORY_COLORS[category],
            label=CATEGORY_LABELS[category],
        )

    # Bars are drawn bottom-up; invert so PLOS sits at the top and each group
    # reads Total -> JATS XML downward, matching the funnel.
    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, df[CATEGORIES].to_numpy().max() * 1.08)
    ax.set_yticks(centers)
    ax.set_yticklabels(JOURNAL_LABELS)
    ax.set_xlabel("Paper Count", fontsize=XY_LABEL_FONT_SIZE)
    ax.set_ylabel("Megajournal", fontsize=XY_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
    ax.grid(visible=False)

    # Title sits directly above the subtitle; the legend goes below the axes.
    ax.set_title("Paper Counts by Megajournal", fontsize=SUPTITLE_FONT_SIZE, pad=36)
    total: int = int(df["Total Papers"].sum())
    candidates: int = int(df["JATS XML Documents"].sum())
    ax.text(
        0.5,
        1.015,
        f"Total papers: {total:,} | Candidate papers: {candidates:,}",
        transform=ax.transAxes,
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="bottom",
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="{:,.0f}",
            padding=3,
            fontsize=BAR_LABEL_FONT_SIZE,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=OTHER_FONT_SIZE,
    )

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
    default=Path("figQ.pdf").absolute(),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output path for the plot.",
)
def main(db_path: Path, output_path: Path) -> None:
    db_path = db_path.absolute()
    output_path = output_path.absolute()
    db: Engine = create_engine(url=f"sqlite:///{db_path}")

    papers: DataFrame = get_papers_per_journal(db=db)
    openalex_papers: DataFrame = get_openalex_papers_per_journal(db=db)
    papers_with_citations: DataFrame = get_papers_with_citations_per_journal(db=db)
    natural_science_papers: DataFrame = get_natural_science_papers_per_journal(db=db)
    jats_papers: DataFrame = get_jats_per_journal(db=db)

    df: DataFrame = create_data(
        df1=papers,
        df2=openalex_papers,
        df3=papers_with_citations,
        df4=natural_science_papers,
        df5=jats_papers,
    )

    plot(df=df, output_path=output_path)

    print(df.set_index("journal")[CATEGORIES].to_string())
    print()
    print(df[CATEGORIES].sum().to_string())


if __name__ == "__main__":
    main()
