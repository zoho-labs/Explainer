package com.zoho.ml.explainer;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;

public final class SparkUtils {

  private static volatile SparkUtils instance;
  private static JavaSparkContext jsc;
  private static SQLContext sqlContext;

  private SparkUtils() {
    if (instance != null) {
      throw new IllegalStateException("Already initialized."); // No I18N
    }
  }

  public static SparkUtils getInstance() {
    SparkUtils result = instance;
    if (result == null) {
      synchronized (SparkUtils.class) {
        result = instance;
        if (result == null) {
          instance = result = new SparkUtils();
          SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("Explainer");
          jsc = new JavaSparkContext(sparkConf);
          sqlContext = new SQLContext(jsc);
        }
      }
    }
    return result;
  }

  public SQLContext getSQLContext() {
    return sqlContext;
  }

  public JavaSparkContext getJavaSparkContext() {
    return jsc;
  }
}
