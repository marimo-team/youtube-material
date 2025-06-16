import marimo

__generated_with = "0.13.15"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    return


@app.cell
def _():
    from great_tables import GT
    from great_tables.data import sp500

    # Define the start and end dates for the data range
    start_date = "2010-06-07"
    end_date = "2010-06-14"

    # Filter sp500 using Pandas to dates between `start_date` and `end_date`
    sp500_mini = sp500[(sp500["date"] >= start_date) & (sp500["date"] <= end_date)]

    # Create a gt table based on the `sp500_mini` table data
    (
        GT(sp500_mini)
        .tab_header(title="S&P 500", subtitle=f"{start_date} to {end_date}")
        .fmt_currency(columns=["open", "high", "low", "close"])
        .fmt_date(columns="date", date_style="wd_m_day_year")
        .fmt_number(columns="volume", compact=True)
        .cols_hide(columns="adj_close")
    )
    return (GT,)


@app.cell
def _(pl):
    coffee_data = [
        {
            "icon": "grinder.png",
            "product": "Grinder",
            "revenue_dollars": 904500.0,
            "revenue_pct": 0.03,
            "profit_dollars": 567960.0,
            "profit_pct": 0.04,
            "monthly_sales": [521, 494, 596, 613, 667, 748, 765, 686, 607, 594, 568, 751],
        },
        {
            "icon": "moka-pot.png",
            "product": "Moka pot",
            "revenue_dollars": 2045250.0,
            "revenue_pct": 0.07,
            "profit_dollars": 181080.0,
            "profit_pct": 0.01,
            "monthly_sales": [
                4726,
                4741,
                4791,
                5506,
                6156,
                6619,
                6868,
                6026,
                5304,
                4884,
                4648,
                6283,
            ],
        },
        {
            "icon": "cold-brew.png",
            "product": "Cold brew",
            "revenue_dollars": 288750.0,
            "revenue_pct": 0.01,
            "profit_dollars": 241770.0,
            "profit_pct": 0.02,
            "monthly_sales": [244, 249, 438, 981, 1774, 2699, 2606, 2348, 1741, 896, 499, 244],
        },
        {
            "icon": "filter.png",
            "product": "Filter",
            "revenue_dollars": 404250.0,
            "revenue_pct": 0.01,
            "profit_dollars": 70010.0,
            "profit_pct": 0.0,
            "monthly_sales": [
                2067,
                1809,
                1836,
                2123,
                2252,
                2631,
                2562,
                2367,
                2164,
                2195,
                2070,
                2744,
            ],
        },
        {
            "icon": "drip-machine.png",
            "product": "Drip machine",
            "revenue_dollars": 2632000.0,
            "revenue_pct": 0.09,
            "profit_dollars": 1374450.0,
            "profit_pct": 0.09,
            "monthly_sales": [
                2137,
                1623,
                1971,
                2097,
                2580,
                2456,
                2336,
                2316,
                2052,
                1967,
                1837,
                2328,
            ],
        },
        {
            "icon": "aeropress.png",
            "product": "AeroPress",
            "revenue_dollars": 2601500.0,
            "revenue_pct": 0.09,
            "profit_dollars": 1293780.0,
            "profit_pct": 0.09,
            "monthly_sales": [
                6332,
                5199,
                6367,
                7024,
                7906,
                8704,
                8693,
                7797,
                6828,
                6963,
                6877,
                9270,
            ],
        },
        {
            "icon": "pour-over.png",
            "product": "Pour over",
            "revenue_dollars": 846000.0,
            "revenue_pct": 0.03,
            "profit_dollars": 364530.0,
            "profit_pct": 0.02,
            "monthly_sales": [
                1562,
                1291,
                1511,
                1687,
                1940,
                2177,
                2141,
                1856,
                1715,
                1806,
                1601,
                2165,
            ],
        },
        {
            "icon": "french-press.png",
            "product": "French press",
            "revenue_dollars": 1113250.0,
            "revenue_pct": 0.04,
            "profit_dollars": 748120.0,
            "profit_pct": 0.05,
            "monthly_sales": [
                3507,
                2880,
                3346,
                3792,
                3905,
                4095,
                4184,
                4428,
                3279,
                3420,
                3297,
                4819,
            ],
        },
        {
            "icon": "cezve.png",
            "product": "Cezve",
            "revenue_dollars": 2512500.0,
            "revenue_pct": 0.09,
            "profit_dollars": 1969520.0,
            "profit_pct": 0.13,
            "monthly_sales": [
                12171,
                11469,
                11788,
                13630,
                15391,
                16532,
                17090,
                14433,
                12985,
                12935,
                11598,
                15895,
            ],
        },
        {
            "icon": "chemex.png",
            "product": "Chemex",
            "revenue_dollars": 3137250.0,
            "revenue_pct": 0.11,
            "profit_dollars": 817680.0,
            "profit_pct": 0.06,
            "monthly_sales": [
                4938,
                4167,
                5235,
                6000,
                6358,
                6768,
                7112,
                6249,
                5605,
                6076,
                4980,
                7220,
            ],
        },
        {
            "icon": "scale.png",
            "product": "Scale",
            "revenue_dollars": 3801000.0,
            "revenue_pct": 0.13,
            "profit_dollars": 2910290.0,
            "profit_pct": 0.2,
            "monthly_sales": [
                1542,
                1566,
                1681,
                2028,
                2425,
                2549,
                2569,
                2232,
                2036,
                2089,
                1693,
                3180,
            ],
        },
        {
            "icon": "kettle.png",
            "product": "Kettle",
            "revenue_dollars": 756250.0,
            "revenue_pct": 0.03,
            "profit_dollars": 617520.0,
            "profit_pct": 0.04,
            "monthly_sales": [
                1139,
                1023,
                1087,
                1131,
                1414,
                1478,
                1456,
                1304,
                1140,
                1233,
                1193,
                1529,
            ],
        },
        {
            "icon": "espresso-machine.png",
            "product": "Espresso Machine",
            "revenue_dollars": 8406000.0,
            "revenue_pct": 0.29,
            "profit_dollars": 3636440.0,
            "profit_pct": 0.25,
            "monthly_sales": [686, 840, 618, 598, 2148, 533, 797, 996, 1002, 668, 858, 2577],
        },
        {
            "icon": None,
            "product": "Total",
            "revenue_dollars": 29448500.0,
            "revenue_pct": 1.0,
            "profit_dollars": 14793150.0,
            "profit_pct": 1.0,
            "monthly_sales": None,
        },
    ]

    coffee_df = pl.DataFrame(coffee_data).drop("icon")
    return (coffee_df,)


@app.cell
def _(GT, coffee_df, pl):
    import polars.selectors as cs
    from great_tables import loc, style

    sel_rev = cs.starts_with("revenue")
    sel_prof = cs.starts_with("profit")


    coffee_table = (
        GT(coffee_df)
        .tab_header("Sales of Coffee Equipment")
        .tab_spanner(label="Revenue", columns=sel_rev)
        .tab_spanner(label="Profit", columns=sel_prof)
        .cols_label(
            revenue_dollars="Amount",
            profit_dollars="Amount",
            revenue_pct="Percent",
            profit_pct="Percent",
            monthly_sales="Monthly Sales",
            product="Product",
        )
        .fmt_number(
            columns=cs.ends_with("dollars"),
            compact=True,
            pattern="${x}",
            n_sigfig=3,
        )
        .fmt_percent(columns=cs.ends_with("pct"), decimals=0)
        # style ----
        .tab_style(
            style=style.fill(color="aliceblue"),
            locations=loc.body(columns=sel_rev),
        )
        .tab_style(
            style=style.fill(color="papayawhip"),
            locations=loc.body(columns=sel_prof),
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(rows=pl.col("product") == "Total"),
        )
        .fmt_nanoplot("monthly_sales", plot_type="bar")
        .sub_missing(missing_text="")
    )

    coffee_table
    return


@app.cell
def _():
    from great_tables import nanoplot_options
    import polars as pl

    races = ["Blood Elf", "Troll", "Tauren", "Orc", "Undead"]

    wow_df = (
        pl.read_parquet("wow-full.parquet")
        .filter(pl.col("race").is_in(races))
        .show()
        .with_columns(date=pl.col("datetime").dt.truncate("4w"))
        .group_by("race", "date")
        .agg(hours=pl.len() / 6, unique_players=pl.n_unique("player_id"))
        .show()
        .group_by("race")
        .agg(
            hours=pl.sum("hours").round().cast(pl.Int32), 
            over_time=pl.col("hours")
        )
        .show()
    )

    (
        wow_df.style
            .tab_header("World of Warcraft stats")
            .fmt_nanoplot(
                "over_time", 
                plot_type="line", 
                options=nanoplot_options(show_data_points=False, show_data_area=False)
            )
    )
    return pl, races


@app.cell
def _(pl):
    def show(self, n=5, name=None):
        if name:
            print(name)
        if isinstance(self, pl.DataFrame):
            print(self.head(n))
        else:
            print(self.head(n).collect())
        return self

    pl.DataFrame.show = show
    pl.LazyFrame.show = show
    return


@app.cell
def _(pl, races):
    _ = (
        pl.scan_parquet("wow-full.parquet")
        .filter(pl.col("race").is_in(races))
        .show()
        .with_columns(date=pl.col("datetime").dt.truncate("4w"))
        .group_by("race", "date")
        .agg(hours=pl.len() / 6, unique_players=pl.n_unique("player_id"))
        .show()
        .group_by("race")
        .agg(
            hours=pl.sum("hours").round().cast(pl.Int32), 
            over_time=pl.col("hours")
        )
        .show()
    )
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
