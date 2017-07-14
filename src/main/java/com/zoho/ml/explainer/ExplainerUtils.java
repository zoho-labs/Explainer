package com.zoho.ml.explainer;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class ExplainerUtils {

  public static List<List<Double>> discretize(List<List<Double>> list,
      List<List<Double>> percentileList) {

    List<Double> discrete = new ArrayList<>();
    int numRows = list.get(0).size();
    int in = 0;
    double[] a;
    for (int j = 0; j < list.size(); j++) {
      a = ExplainerUtils.listToDoubleArray(percentileList.get(j));
      for (int i = 0; i < numRows; i++) {
        in = Arrays.binarySearch(a, list.get(j).get(i));
        if (in < 0) {
          in = (-in) - 1;
        }
        discrete.add(Double.valueOf(in));
      }
    }
    return ExplainerUtils.getSubLists(discrete, numRows);

  }

  public static List<List<Double>> undiscretize(List<List<Double>> discretizedContinuous,
      List<List<Double>> continuousList, List<List<Double>> sampleList, int[] percentileValues) {

    List<Double> selection, min, max, val;
    double[] array;
    SecureRandom random;
    int s;

    List<Double> mean = new ArrayList<>();
    List<Double> std = new ArrayList<>();
    for (int q = 0; q < discretizedContinuous.size(); q++) {
      for (int i = 0; i < (percentileValues.length + 1); i++) {
        selection = new ArrayList<>();
        for (int j = 0; j < discretizedContinuous.get(0).size(); j++) {
          if (i == discretizedContinuous.get(q).get(j)) {
            selection.add(continuousList.get(q).get(j));
          }
        }
        if (selection.size() != 0) {
          array = listToDoubleArray(selection);
          mean.add(StatUtils.mean(array));
          std.add(Math.sqrt(StatUtils.variance(array)));
        } else {
          mean.add(0.0);
          std.add(0.0);
        }
      }
    }

    List<List<Double>> listPercentiles = calculatePercentiles(continuousList, percentileValues);
    List<List<Double>> listBoundaries = calculateBoundaries(continuousList);
    List<List<Double>> minLists = new ArrayList<>();
    List<List<Double>> maxLists = new ArrayList<>();
    for (int i = 0; i < continuousList.size(); i++) {
      min = new ArrayList<>();
      min.add(listBoundaries.get(i).get(0));
      for (int j = 0; j < percentileValues.length; j++) {
        min.add(listPercentiles.get(i).get(j));
      }
      minLists.add(min);
    }
    for (int i = 0; i < continuousList.size(); i++) {
      max = new ArrayList<>();
      for (int j = 0; j < percentileValues.length; j++) {
        max.add(listPercentiles.get(i).get(j));
      }
      max.add(listBoundaries.get(i).get(1));
      maxLists.add(max);
    }

    List<List<Double>> meanLists = getSubLists(mean, (percentileValues.length + 1));
    List<List<Double>> stdLists = getSubLists(std, (percentileValues.length + 1));
    List<Double> undiscretized = new ArrayList<>();
    for (int i = 0; i < sampleList.size(); i++) {
      for (int j = 0; j < sampleList.get(0).size(); j++) {
        s = sampleList.get(i).get(j).intValue();
        random = new SecureRandom();
        val = new ArrayList<>();
        val.add(minLists.get(i).get(s));
        val.add(Math.round((random.nextGaussian() * stdLists.get(i).get(s))
            + meanLists.get(i).get(s) * 100.0) / 100.0);
        val.add(maxLists.get(i).get(s));
        undiscretized.add(Collections.max(val));
      }
    }
    return getSubLists(undiscretized, sampleList.get(0).size());

  }

  public static List<List<Double>> dataWithSampleWeights(List<Double> weights,
      List<List<Double>> weightLists, String delimiter) {

    List<Double> weightedRows, wtdDatapoints;
    double weightedDatapoints;
    int i;
    Row[] rows = dataframeFromList(weightLists, delimiter).collect();

    List<Double> avgList = new ArrayList<>();
    for (i = 0; i < weightLists.size(); i++) {
      avgList.add(0.0);
    }

    List<List<Double>> weightedData = new ArrayList<>();
    for (i = 0; i < weights.size(); i++) {
      weightedRows = new ArrayList<>();
      for (int j = 0; j < weightLists.size(); j++) {
        weightedDatapoints = Double.valueOf(rows[i].get(j).toString()) * (weights.get(i));
        weightedRows.add(weightedDatapoints);
        avgList.set(j, (avgList.get(j) + weightedDatapoints));
      }
      weightedData.add(weightedRows);
    }

    double sumOfWeights = 0.0;
    for (i = 0; i < weights.size(); i++) {
      sumOfWeights = sumOfWeights + weights.get(i);
    }
    for (i = 0; i < avgList.size(); i++) {
      avgList.set(i, avgList.get(i) / sumOfWeights);
    }

    List<Double> sqrtWts = new ArrayList<>();
    for (i = 0; i < weights.size(); i++) {
      sqrtWts.add(Math.sqrt(weights.get(i)));
    }

    List<List<Double>> wtdData = new ArrayList<>();
    for (i = 0; i < weightLists.get(0).size(); i++) {
      wtdDatapoints = new ArrayList<>();
      for (int j = 0; j < weightLists.size(); j++) {
        wtdDatapoints.add((weightLists.get(j).get(i) - avgList.get(j)) * sqrtWts.get(i));
      }
      wtdData.add(wtdDatapoints);
    }
    return wtdData;

  }

  public static List<List<Double>> replaceContinuousSamples(List<List<Double>> list,
      List<List<Double>> newList, List<Integer> categoricalFeatures) {

    int l = 0;
    int k = 0;

    List<List<Double>> replacedList = new ArrayList<>();
    for (int i = 0; i < list.size(); i++) {
      if (categoricalFeatures.size() == 0) {
        replacedList.add(newList.get(i));
      } else {
        if (i == (categoricalFeatures.get(l) - 1)) {
          replacedList.add(list.get(i));
          if (l < (categoricalFeatures.size() - 1)) {
            l++;
          }
        } else {
          replacedList.add(newList.get(k));
          k++;
        }
      }
    }
    return replacedList;

  }

  public static List<List<Double>> replaceCategoricalSamples(List<List<Double>> list,
      List<List<Double>> newList, List<Integer> categoricalFeatures) {

    int l = 0;

    List<List<Double>> replacedList = new ArrayList<>();
    for (int i = 0; i < list.size(); i++) {
      if (categoricalFeatures.size() == 0) {
        replacedList.add(list.get(i));
      } else {
        if (i == (categoricalFeatures.get(l) - 1)) {
          replacedList.add(newList.get(l));
          if (l < (categoricalFeatures.size() - 1)) {
            l++;
          }
        } else {
          replacedList.add(list.get(i));
        }
      }
    }
    return replacedList;

  }

  public static List<Double> findMean(List<List<Double>> list) {

    List<Double> mean = new ArrayList<>();
    for (List<Double> l : list) {
      mean.add(StatUtils.mean(listToDoubleArray(l)));
    }
    return mean;

  }

  public static List<Double> findStdDev(List<List<Double>> list) {

    List<Double> stdDev = new ArrayList<>();
    for (List<Double> l : list) {
      stdDev.add(FastMath.sqrt(StatUtils.variance(listToDoubleArray(l))));
    }
    return stdDev;

  }

  public static List<List<Double>> calculatePercentiles(List<List<Double>> list, int[] percentiles) {

    double value = 0.0;

    List<Double> percentile = new ArrayList<>();
    for (List<Double> l : list) {
      for (int p : percentiles) {
        value = new Percentile().evaluate(listToDoubleArray(l), p);
        value = Math.round(value * 100.0) / 100.0;
        percentile.add(value);
      }
    }
    return getSubLists(percentile, percentiles.length);

  }

  public static List<List<Double>> calculateBoundaries(List<List<Double>> list) {

    List<Double> boundaries = new ArrayList<>();
    for (List<Double> l : list) {
      boundaries.add(Math.round(Collections.min(l) * 100.0) / 100.0);
      boundaries.add(Math.round(Collections.max(l) * 100.0) / 100.0);
    }
    return getSubLists(boundaries, 2);

  }

  public static List<Double> randomSamplingFromNormal(double meanValue, double stdValue,
      int numberOfSamples) {

    SecureRandom random = new SecureRandom();
    List<Double> randomSamples = new ArrayList<Double>();
    for (int j = 0; j < numberOfSamples; j++) {
      randomSamples.add((random.nextGaussian() * stdValue) + meanValue);
    }
    return randomSamples;

  }

  public static List<Double> weightedSamplingWithReplacement(List<Double> values,
      List<Double> weights, int numberOfSamples) {

    int minIndex, maxIndex, sampleIndex;
    int size = values.size();
    // Calculating the next power of two
    int power = (int) Math.ceil(Math.log((double) size) / Math.log(2));
    int numPartitions = (int) Math.pow(2, power);
    double minValue, maxValue, remainingCapacity, sample, w, elementWeight;
    double capacity = 1.00 / numPartitions;
    List<Double> partition, samplePartition;
    SecureRandom random = new SecureRandom();

    List<List<Double>> listPartitions = new ArrayList<>(numPartitions);
    for (int i = 0; i < numPartitions; i++) {
      partition = new ArrayList<Double>();
      minValue = Collections.min(weights);
      minIndex = weights.indexOf(minValue);
      partition.add((double) minIndex);
      if (minValue >= capacity) {
        weights.set(minIndex, (minValue - capacity));
        partition.add(-1.00);
        partition.add(1.00);
      } else {
        remainingCapacity = capacity - minValue;
        maxValue = Collections.max(weights);
        maxIndex = weights.indexOf(maxValue);
        partition.add((double) maxIndex);
        partition.add(minValue / capacity);
        weights.set(minIndex, 0.0);
        weights.set(maxIndex, (maxValue - remainingCapacity));
      }
      listPartitions.add(partition);
    }

    List<Double> samples = new ArrayList<>();
    for (int i = 0; i < numberOfSamples; i++) {
      sample = random.nextDouble() * numPartitions;
      sampleIndex = (int) sample;
      w = sample - sampleIndex;
      samplePartition = listPartitions.get(sampleIndex);
      elementWeight = samplePartition.get(2);
      if (w <= elementWeight) {
        samples.add(values.get(samplePartition.get(0).intValue()));
      } else {
        samples.add(values.get(samplePartition.get(1).intValue()));
      }
    }
    return samples;

  }

  public static JavaRDD<LabeledPoint> convertRDDStringToLabeledPoint(JavaRDD<String> data,
      final String delimiter) {
    JavaRDD<LabeledPoint> labeledPointData = data.map(new Function<String, LabeledPoint>() {
      private static final long serialVersionUID = 1L;

      public LabeledPoint call(String data) throws Exception {
        String splitter[] = data.split(delimiter);
        double[] array = new double[splitter.length - 1];
        for (int i = 0; i < array.length; i++) {
          try {
            array[i] = Double.parseDouble(splitter[i + 1]);
          } catch (Exception e) {
            throw new Exception(this.getClass() + " Cannot convert \"" + splitter[i + 1]
                + "\" to double");
          }
        }
        return new LabeledPoint(Double.parseDouble(splitter[0]), Vectors.dense(array));
      }
    });
    return labeledPointData;
  }

  public static List<List<Double>> constructListWithColumnNames(DataFrame dataframe,
      String[] columnNames) {

    List<Double> l;
    Row[] rows;

    List<List<Double>> list = new ArrayList<>();
    for (String name : columnNames) {
      l = new ArrayList<>();
      rows = dataframe.select(name).collect();
      for (Row r : rows) {
        l.add(Double.valueOf(r.get(0).toString()));
      }
      list.add(l);
    }
    return list;

  }

  public static List<String> getAppendedList(List<List<Double>> list, String delimiter) {

    StringBuilder builder;
    String str;

    List<String> appendedList = new ArrayList<String>();
    for (int i = 0; i < list.get(0).size(); i++) {
      builder = new StringBuilder();
      str = "";
      for (int j = 0; j < list.size(); j++) {
        builder.append(list.get(j).get(i)).append(delimiter);
      }
      str = builder.toString();
      appendedList.add(str.substring(0, str.lastIndexOf(delimiter)));
    }
    return appendedList;

  }

  public static DataFrame dataframeFromList(List<List<Double>> list, String delimiter) {

    JavaRDD<String> data =
        SparkUtils.getInstance().getJavaSparkContext()
            .parallelize(ExplainerUtils.getAppendedList(list, delimiter));
    JavaRDD<Row> rawData = data.map(new Function<String, Row>() {
      private static final long serialVersionUID = 1L;

      public Row call(String data) {
        Row newRow = RowFactory.create(data);
        Object[] colArray = (String[]) newRow.getString(0).split(Pattern.quote(delimiter));
        return RowFactory.create(colArray);
      }
    });
    StructField[] structField = new StructField[rawData.first().size()];
    StringBuilder builder;
    for (int i = 0; i < structField.length; i++) {
      builder = new StringBuilder("C").append(i + 1);
      structField[i] =
          new StructField(builder.toString(), DataTypes.StringType, false, Metadata.empty());
    }
    StructType schema = new StructType(structField);
    return (SparkUtils.getInstance().getSQLContext().createDataFrame(rawData, schema));

  }

  public static List<List<Double>> getSubLists(List<Double> list, int size) {

    List<List<Double>> listOfLists = new LinkedList<List<Double>>();
    for (int i = 0; i < list.size(); i += size) {
      listOfLists.add(new ArrayList<Double>(list.subList(i, Math.min(i + size, list.size()))));
    }
    return listOfLists;

  }

  public static double[] listToDoubleArray(List<Double> list) {

    double[] array = new double[list.size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = list.get(i).doubleValue();
    }
    return array;

  }

  public static List<List<Double>> dataframeToList(DataFrame dataframe) {

    List<Double> column;
    Row[] rows;

    List<List<Double>> listOfColumns = new ArrayList<>();
    for (String s : dataframe.columns()) {
      column = new ArrayList<>();
      rows = dataframe.select(s).collect();
      for (Row r : rows) {
        column.add(Double.valueOf(r.get(0).toString()));
      }
      listOfColumns.add(column);
    }
    return listOfColumns;

  }
}
