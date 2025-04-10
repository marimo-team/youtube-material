import polars as pl 

def read_data():
    return pl.read_parquet("~/Downloads/wow-full.parquet")

def set_types(dataf):
    return (
        dataf.with_columns([
            pl.col("guild").is_not_null(),
            pl.col("datetime").cast(pl.Int64).alias("timestamp")
        ])
    )

def clean_data(dataf):
    return (
        dataf
        .filter(
            ~pl.col("class").is_in(["482", "Death Knight", "3485伊", "2400"]),
            pl.col("race").is_in(["Troll", "Orc", "Undead", "Tauren", "Blood Elf"])
        )
    )

def sessionize(dataf, threshold=20 * 60 * 1000):
    return (dataf
                .sort(["player_id", "timestamp"])
                .with_columns(
                    (pl.col("timestamp").diff().cast(pl.Int64) > threshold).fill_null(True).alias("ts_diff"),
                    (pl.col("player_id").diff() != 0).fill_null(True).alias("char_diff"),
                )
                .with_columns(
                    (pl.col("ts_diff") | pl.col("char_diff")).alias("new_session_mark")
                )
                .with_columns(
                    pl.col("new_session_mark").cum_sum().alias("session")   
                )
                .drop(["char_diff", "ts_diff", "new_session_mark"]))

def add_features(dataf):
    return (dataf
                .with_columns(
                    pl.col("player_id").count().over("session").alias("session_length"),
                    pl.col("session").n_unique().over("player_id").alias("n_sessions_per_char")
                ))

def remove_bots(dataf, max_session_hours=24):
    # We're using some domain knowledge. The logger of our dataset should log
    # data every 10 minutes. That's what this line is based on.
    n_rows = max_session_hours * 6
    return (dataf
            .filter(pl.col("session_length").max().over("player_id") < n_rows))
