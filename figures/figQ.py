from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame, Series
from sqlalchemy import Engine, create_engine

SUPTITLE_FONT_SIZE: int = 24
TITLE_FONT_SIZE: int = 22
XY_LABEL_FONT_SIZE: int = 20
XY_TICK_FONT_SIZE: int = 18
OTHER_FONT_SIZE: int = 18
FIGSIZE: tuple[float, float] = (12.8, 9.6)


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

    merged_df = pd.merge(all_papers_df, doi_year_df, on="doi", how="left")

    merged_df["publication_year"] = merged_df["publication_year"].fillna(0).astype(int)

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
    openalex oa
ON
    oa.doi = ns.doi
JOIN
    natural_science_article_dois ns
ON
    ns.doi == a.doi
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
    data: dict[str, list[str | int]] = {
        "journal": ["BMJ", "F1000", "FrontiersIn", "PLOS"],
        "Total Papers": [],
        "OpenAlex Indexed Papers": [],
        "Papers With Citations": [],
        "Natural Science Papers": [],
        "JATS XML Documents": [],
    }

    def _run(key: str, df: DataFrame) -> None:
        counts: Series = df["megajournal"].value_counts()
        data[key].append(counts["BMJ"])
        data[key].append(counts["F1000"])
        data[key].append(counts["FrontiersIn"])
        data[key].append(counts["PLOS"])

    _run("Total Papers", df1)
    _run("OpenAlex Indexed Papers", df2)
    _run("Papers With Citations", df3)
    _run("Natural Science Papers", df4)
    _run("JATS XML Documents", df5)

    return DataFrame(data=data)


def plot(df: DataFrame, output_path: Path) -> None:
    # Convert wide → long format
    df_long = df.melt(
        id_vars="journal",
        value_vars=[
            "Total Papers",
            "OpenAlex Indexed Papers",
            "Papers With Citations",
            "Natural Science Papers",
            "JATS XML Documents",
        ],
        var_name="category",
        value_name="count",
    )

    # Plot grouped bar chart
    plt.figure(figsize=FIGSIZE)
    ax = sns.barplot(
        data=df_long,
        x="journal",
        y="count",
        hue="category",
        # log_scale=True,
    )

    # ---- Y-AXIS: commas + headroom ----
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, df_long["count"].max() * 1.15)  # 15% headroom

    plt.suptitle("Paper Counts by Megajournal", fontsize=SUPTITLE_FONT_SIZE)
    plt.title(
        label=f"{df['Total Papers'].sum():,} Total Papers; {df['JATS XML Documents'].sum():,} Candidate Papers",
        fontsize=TITLE_FONT_SIZE,
        loc="center",
    )
    plt.xlabel("Megajournal", fontsize=XY_LABEL_FONT_SIZE)
    plt.ylabel("Paper Count", fontsize=XY_LABEL_FONT_SIZE)
    plt.yticks(fontsize=XY_TICK_FONT_SIZE)
    plt.xticks(fontsize=XY_TICK_FONT_SIZE)
    plt.legend(title="", fontsize=OTHER_FONT_SIZE)

    # ---- ADD VALUE LABELS ----
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="{:,.0f}",
            padding=3,
            fontsize=OTHER_FONT_SIZE,
            rotation=60,
        )

    plt.tight_layout()
    plt.savefig(output_path)


@click.command()
@click.option(
    "--db",
    "db_path",
    default=Path("../data/aius.3-18-2026.db").resolve(),
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

    print(df.sum())


if __name__ == "__main__":
    main()
