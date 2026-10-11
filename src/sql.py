"""Build data/oil.duckdb from the raw files, and read it back.

    python -m src.sql

Two tables. demand holds the three targets, one row per product and month. predictors holds
every raw driver in long form, one row per product, variable and period_end: the last day of
the period the number describes, not the day it was published.
"""

import duckdb

from src import config as c

US = [("gasoline", c.GASOLINE), ("distillate", c.DISTILLATE)]


def query(sql):
    with duckdb.connect(c.DB, read_only=True) as con:
        return con.sql(sql).df()


def build():
    c.DB.unlink(missing_ok=True)
    with duckdb.connect(c.DB) as con:
        con.execute("CREATE TABLE demand (product VARCHAR, period_end DATE, value DOUBLE, "
                    "unit VARCHAR)")
        con.execute("CREATE TABLE predictors (product VARCHAR, variable VARCHAR, period_end DATE, "
                    "value DOUBLE)")

        # US: EIA monthly files are dated mid-month, weekly files on the Friday that ends the week
        for product, folder in US:
            con.execute(f"""
                INSERT INTO demand
                SELECT '{product}', last_day(date), product_supplied, 'kb/d'
                FROM read_csv('{folder}/eia_monthly.csv')
                WHERE product_supplied IS NOT NULL""")
            con.execute(f"""
                INSERT INTO predictors
                SELECT '{product}', variable, last_day(date), value
                FROM (UNPIVOT read_csv('{folder}/eia_monthly.csv')
                      ON COLUMNS(* EXCLUDE (date, product_supplied))
                      INTO NAME variable VALUE value)""")
            con.execute(f"""
                INSERT INTO predictors
                SELECT '{product}', variable, date, value
                FROM (UNPIVOT (SELECT date, product_supplied AS product_supplied_weekly, stocks
                               FROM read_csv('{folder}/eia_weekly.csv'))
                      ON product_supplied_weekly, stocks INTO NAME variable VALUE value)""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'distillate', variable, last_day(CAST(month || '-01' AS DATE)), value
            FROM (UNPIVOT read_csv('{c.DISTILLATE}/noaa_hdd.csv') ON hdd, hdd_oil
                  INTO NAME variable VALUE value)""")

        # FRED files are wide and dated on the first day of the month
        for product, folder in US + [("bunker", c.BUNKER)]:
            con.execute(f"""
                INSERT INTO predictors
                SELECT '{product}', variable, last_day(date), value
                FROM (UNPIVOT read_csv('{folder}/fred.csv') ON COLUMNS(* EXCLUDE (date))
                      INTO NAME variable VALUE value)""")

        # Singapore: monthly sales in kt summed over the fuel-oil grades (HSFO, LSFO, ULSFO and
        # their bio-blends; gasoil, LNG and the other alternatives are left out), then divided by
        # the days in the month, so February is not a demand drop and the unit is a daily rate
        # like EIA's
        con.execute(f"""
            INSERT INTO demand
            SELECT 'bunker', last_day(CAST(month || '-01' AS DATE)) AS period_end,
                   SUM(bunker_sales) / day(period_end), 'kt/d'
            FROM read_csv('{c.BUNKER}/mpa_bunker_sales.csv')
            WHERE bunker_type LIKE '%Fuel Oil'
            GROUP BY month""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'bunker', variable, last_day(CAST(month || '-01' AS DATE)), value
            FROM (UNPIVOT (SELECT month, number_of_vessel_calls AS bunker_calls,
                                  gross_tonnage AS bunker_calls_gt
                           FROM read_csv('{c.BUNKER}/mpa_vessel_calls.csv')
                           WHERE purpose_type = 'Bunkers')
                  ON bunker_calls, bunker_calls_gt INTO NAME variable VALUE value)""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'bunker', 'cargo', last_day(CAST(month || '-01' AS DATE)), cargo_throughput
            FROM read_csv('{c.BUNKER}/mpa_cargo.csv')
            UNION ALL
            SELECT 'bunker', 'tankers', last_day(CAST(month || '-01' AS DATE)), number_of_tankers
            FROM read_csv('{c.BUNKER}/mpa_tankers.csv')
            UNION ALL
            SELECT 'bunker', 'wtr', last_day(CAST(month || '-01' AS DATE)), wtr
            FROM read_csv('{c.BUNKER}/cpb_world_trade.csv')""")

        print(con.sql("SELECT product, unit, MIN(period_end), MAX(period_end), COUNT(*) "
                      "FROM demand GROUP BY 1, 2 ORDER BY 1"))
        print(con.sql("SELECT product, variable, MIN(period_end), MAX(period_end), COUNT(*) "
                      "FROM predictors GROUP BY 1, 2 ORDER BY 1, 2"))


if __name__ == "__main__":
    build()
