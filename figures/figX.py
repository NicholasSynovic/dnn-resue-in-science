from json import loads
from pathlib import Path
from textwrap import fill
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
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
FIGSIZE: tuple[float, float] = (24, 10)


ADAPTATION_LABEL: str = "Adaptation Reuse"
CONCEPTUAL_LABEL: str = "Conceptual Reuse"
DEPLOYMENT_LABEL: str = "Deployment Reuse"
CLASS_ORDER: list[str] = [
    ADAPTATION_LABEL,
    CONCEPTUAL_LABEL,
    DEPLOYMENT_LABEL,
]


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
JOIN
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
    except Exception:
        return None

    if isinstance(parsed, dict) and parsed:
        return parsed

    return None


def extract_classification(obj: Any) -> str | None:
    if isinstance(obj, dict):
        value = obj.get("classification")
        return value if isinstance(value, str) and value else None

    return None


def create_field_dataframes(df: DataFrame) -> dict[str, DataFrame]:
    data: dict[str, list[str | int]] = {
        "year": [],
        "field": [],
        "classification": [],
    }

    for _, row in df.sort_values(by="publication_year").iterrows():
        if int(row["publication_year"]) < 2012:
            continue

        parsed_response = parse_json(str(row["model_response"]))
        classification = extract_classification(parsed_response)
        if classification is None:
            continue

        if classification not in {
            "adaptation_reuse",
            "conceptual_reuse",
            "deployment_reuse",
        }:
            continue

        topics: list[str] = [
            str(row["topic_0"]),
            str(row["topic_1"]),
            str(row["topic_2"]),
        ]

        for topic in topics:
            if topic in FIELD:
                data["year"].append(int(row["publication_year"]))
                data["field"].append(topic)
                data["classification"].append(classification)

    data_df = DataFrame(data=data)
    if data_df.empty:
        empty_columns = ["year"] + CLASS_ORDER
        return {field: DataFrame(columns=empty_columns) for field in FIELD}

    min_year = 2012
    max_year = int(data_df["year"].max())
    years = list(range(min_year, max_year + 1))
    classification_map = {
        "adaptation_reuse": ADAPTATION_LABEL,
        "conceptual_reuse": CONCEPTUAL_LABEL,
        "deployment_reuse": DEPLOYMENT_LABEL,
    }
    category_order = [
        classification_map[key]
        for key in ["adaptation_reuse", "conceptual_reuse", "deployment_reuse"]
    ]

    field_dataframes: dict[str, DataFrame] = {}
    for field in FIELD:
        field_df = data_df.loc[data_df["field"] == field].copy()
        field_df["classification"] = field_df["classification"].replace(
            classification_map
        )

        if field_df.empty:
            field_dataframes[field] = DataFrame(
                {
                    "year": years,
                    ADAPTATION_LABEL: [0] * len(years),
                    CONCEPTUAL_LABEL: [0] * len(years),
                    DEPLOYMENT_LABEL: [0] * len(years),
                }
            )
            continue

        field_counts = (
            field_df.groupby(["year", "classification"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        full_index = pd.MultiIndex.from_product(
            [years, category_order],
            names=["year", "classification"],
        )
        field_counts = (
            field_counts.set_index(["year", "classification"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        pivot = (
            field_counts.pivot(index="year", columns="classification", values="count")
            .reindex(years, fill_value=0)
            .reset_index()
        )

        for label in category_order:
            if label not in pivot.columns:
                pivot[label] = 0

        pivot = pivot[["year"] + category_order]
        pivot[category_order] = pivot[category_order].astype(int)
        field_dataframes[field] = pivot

    return field_dataframes


def plot(field_dataframes: dict[str, DataFrame], output_path: Path) -> None:
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

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=FIGSIZE, sharey="row")
    flat_axes = axes.flatten()

    row_max: list[int] = []
    for row_fields in [FIELD[:4], FIELD[4:]]:
        row_max.append(
            max(
                (
                    int(field_dataframes[field][CLASS_ORDER].sum(axis=1).max())
                    for field in row_fields
                ),
                default=0,
            )
        )

    colors = {
        ADAPTATION_LABEL: "#4C78A8",
        CONCEPTUAL_LABEL: "#54A24B",
        DEPLOYMENT_LABEL: "#C44E52",
    }

    for index, field in enumerate(FIELD):
        ax = flat_axes[index]
        panel_data: DataFrame = field_dataframes[field]
        wrapped_title = fill(field, width=30) if len(field) > 30 else field
        row_index = index // 4

        if panel_data.empty:
            panel_data = DataFrame(
                {
                    "year": years,
                    ADAPTATION_LABEL: [0] * len(years),
                    CONCEPTUAL_LABEL: [0] * len(years),
                    DEPLOYMENT_LABEL: [0] * len(years),
                }
            )

        ax.bar(
            panel_data["year"],
            panel_data["Adaptation Reuse"],
            # bottom=bottom,
            color=colors["Adaptation Reuse"],
            label="Adaptation Reuse",
        )

        ax.bar(
            panel_data["year"],
            panel_data["Deployment Reuse"],
            # bottom=bottom,
            color=colors["Deployment Reuse"],
            label="Deployment Reuse",
        )

        ax.bar(
            panel_data["year"],
            panel_data["Conceptual Reuse"],
            # bottom=bottom,
            color=colors["Conceptual Reuse"],
            label="Conceptual Reuse",
        )

        # bottom = pd.Series(0, index=panel_data.index)
        # for classification in CLASS_ORDER:
        #     ax.bar(
        #         panel_data["year"],
        #         panel_data[classification],
        #         bottom=bottom,
        #         color=colors[classification],
        #         label=classification,
        #     )
        #     bottom = bottom + panel_data[classification]

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylim(0, row_max[row_index] * 1.3 if row_max[row_index] else 1)
        ax.set_title(wrapped_title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Year" if row_index == 1 else "", fontsize=XY_LABEL_FONT_SIZE)
        ax.set_ylabel(
            "Paper Count" if index in (0, 4) else "", fontsize=XY_LABEL_FONT_SIZE
        )

        years: list[int] = list(range(2012, int(panel_data["year"].max()) + 1))
        ax.set_xticks(years)
        ax.set_xticklabels(
            [
                str(year) if position % 2 == 0 else ""
                for position, year in enumerate(years)
            ]
        )

        ax.tick_params(axis="both", labelsize=XY_TICK_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45)

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
                    Patch(color=colors[classification], label=classification)
                    for classification in CLASS_ORDER
                ],
                loc="center left",
                frameon=True,
                fontsize=OTHER_FONT_SIZE,
            )

    fig.suptitle(
        "PTM Reuse Classification Counts per Year by Field",
        fontsize=SUPTITLE_FONT_SIZE,
    )
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
    default=Path("figX.pdf").absolute(),
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
    plot(field_dataframes=field_dataframes, output_path=output_path)


if __name__ == "__main__":
    main()
