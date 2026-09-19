"""Pipeline Tier 2 (Primary): Spark Structured Streaming.

raw-trades (Kafka) -> event-time dedup + 5s bar aggregation -> bars (Kafka)

Key design choices (defend in paper):
  - event-time (T field), NOT processing time -> watermark handles late data
  - dropDuplicates([s, a]) -> streaming dedup within watermark window
  - open/close via min/max(struct(ts, p)) -> ordering-safe without sorting
  - append outputMode -> bars emitted after watermark passes (safe, no retract)

Run:
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      src/pipeline/spark_streaming.py

Or via convenience wrapper:
    python -m src.pipeline.spark_streaming --config config/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    P, B = cfg["pipeline"], cfg["bars"]

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import (BooleanType, DoubleType, LongType,
                                       StringType, StructField, StructType)
    except ImportError:
        print("PySpark not installed. See README for fallback.")
        return

    SCHEMA = StructType([
        StructField("s", StringType()),
        StructField("a", LongType()),
        StructField("p", DoubleType()),
        StructField("q", DoubleType()),
        StructField("T", LongType()),
        StructField("m", BooleanType()),
    ])

    spark = (SparkSession.builder.appName("pnd-bars")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", P["kafka_bootstrap"])
           .option("subscribe", P["topic_raw"])
           .option("startingOffsets", "latest")
           .load())

    trades = (raw
              .select(F.from_json(F.col("value").cast("string"),
                                  SCHEMA).alias("j"))
              .select("j.*")
              .withColumn("ts", F.timestamp_millis(F.col("T")))
              .withWatermark("ts", "30 seconds")
              .dropDuplicates(["s", "a"]))

    bar_s = B["bar_seconds"]
    bars = (trades
            .groupBy(F.window("ts", f"{bar_s} seconds").alias("w"),
                     F.col("s"))
            .agg(
                F.min(F.struct("ts", "p")).alias("first"),
                F.max(F.struct("ts", "p")).alias("last"),
                F.max("p").alias("high"),
                F.min("p").alias("low"),
                F.sum("q").alias("volume"),
                F.count(F.lit(1)).alias("n_trades"),
                F.sum(F.when(~F.col("m"), F.col("q")).otherwise(0.0))
                 .alias("taker_buy_vol"),
                F.sum(F.col("p") * F.col("q")).alias("quote_vol"),
            )
            .select(
                F.col("s").alias("symbol"),
                F.col("w.start").alias("bar_time"),
                F.col("first.p").alias("open"),
                F.col("high"),
                F.col("low"),
                F.col("last.p").alias("close"),
                "volume", "n_trades", "taker_buy_vol", "quote_vol",
            ))

    out = bars.select(
        F.col("symbol").alias("key"),
        F.to_json(F.struct(
            "symbol", "bar_time", "open", "high", "low", "close",
            "volume", "n_trades", "taker_buy_vol", "quote_vol",
        )).alias("value"))

    ckpt = os.path.join(cfg["paths"]["artifacts_dir"], "spark_ckpt")
    (out.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", P["kafka_bootstrap"])
        .option("topic", P["topic_bars"])
        .option("checkpointLocation", ckpt)
        .outputMode("append")
        .start()
        .awaitTermination())


if __name__ == "__main__":
    main()
