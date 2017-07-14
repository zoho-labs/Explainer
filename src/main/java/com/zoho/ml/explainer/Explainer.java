package com.zoho.ml.explainer;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.RidgeRegressionModel;
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;

public class Explainer implements Serializable {

  private static final long serialVersionUID = 1L;

  private final ExplainerParameters expParams;

  public Explainer(ExplainerParameters expParams) throws Exception {
    if (expParams == null) {
      throw new Exception("ExplainerParameters cannot be null. Specify the necessary parameters.");
    }
    this.expParams = expParams;
  }

  public ExplainerResults explain(String stringQuery) throws Exception {
    DataFrame inputData =
        SparkUtils.getInstance().getSQLContext().read().format("com.databricks.spark.csv")
            .option("inferSchema", "true").option("delimiter", this.expParams.getDelimiter())
            .option("header", String.valueOf(this.expParams.isColumnNameSpecified()))
            .load(this.expParams.getDataPath());

    final String impurity = this.expParams.getImpurity();
    final int maxDepth = this.expParams.getMaxDepth();
    final int maxBins = this.expParams.getMaxBins();
    final int numTrees = this.expParams.getNumTrees();
    final int seed = this.expParams.getSeed();
    final int numClasses = this.expParams.getNumClasses();
    final String featureSubsetStrategy = this.expParams.getFeatureSubsetStrategy();
    final String delimiter = Pattern.quote(this.expParams.getDelimiter());

    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();

    JavaRDD<String> javardd = inputData.toJavaRDD().map(new Function<Row, String>() {
      private static final long serialVersionUID = 1L;

      public String call(Row row) {
        return row.mkString(expParams.getDelimiter());
      }
    });

    JavaRDD<LabeledPoint> data = ExplainerUtils.convertRDDStringToLabeledPoint(javardd, delimiter);

    final RandomForestModel model =
        RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees,
            featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

    String splitter[] = stringQuery.split(delimiter);
    double[] features = new double[splitter.length];
    for (int i = 0; i < splitter.length; i++) {
      splitter[i] = splitter[i].trim();
      if (splitter[i].isEmpty()) {
        throw new Exception(this.getClass() + " : Value missing in " + i
            + " column in the given query \"" + stringQuery + "\"");
      }
      features[i] = Double.parseDouble(splitter[i].trim());
    }
    Vector featureVector = Vectors.dense(features);
    double labelToBeExplained = model.predict(featureVector);

    ExplainerResults expResult = explainerImpl(model, stringQuery, labelToBeExplained, inputData);
    return expResult;
  }

  private ExplainerResults explainerImpl(RandomForestModel model, String stringQuery,
      double labelToBeExplained, DataFrame inputData) {

    // Input Parameters
    // Number of samples
    final int numSamples = this.expParams.getNumSamples();
    // Number of iterations used in ridge reg when 'discretize = false'
    final int numIterations = this.expParams.getNumberOfIterations();
    // Step size used in ridge reg when 'discretize = false
    final double stepSize = this.expParams.getStepSize();
    // Percentiles for discretization
    final int[] percentileValues = this.expParams.getPercentileValues();
    final String delimiter = Pattern.quote(this.expParams.getDelimiter());

    // Removing first column if column name is specified
    String[] actualColNames = inputData.columns();
    StringBuilder builder;
    if (this.expParams.isColumnNameSpecified()) {
      for (int i = 0; i < actualColNames.length; i++) {
        builder = new StringBuilder("C").append(i);
        inputData = inputData.withColumnRenamed(actualColNames[i], builder.toString());
      }
    }

    // Categorical columns
    List<Integer> categoricalColumns = this.expParams.getCategoricalColumns();
    List<Integer> categoricalList = new ArrayList<Integer>();
    if (!(categoricalColumns == null || categoricalColumns.isEmpty())) {
      categoricalList.addAll(categoricalColumns);
    }
    Collections.sort(categoricalList);

    // Removing Label
    String[] colNames = inputData.columns();
    String label = colNames[0];
    DataFrame featuresDataFrame = inputData.drop(label);
    String[] columns = featuresDataFrame.columns();
    int numColumns = columns.length;
    List<List<Double>> featuresList = ExplainerUtils.dataframeToList(featuresDataFrame);

    // Number of features needed for explanation
    int numFeatures = this.expParams.getNumberOfFeatures();
    if (numFeatures <= 0 || numFeatures > numColumns) {
      numFeatures = numColumns;
    }

    // Preparing Query
    List<List<Double>> listQuery;
    DataFrame dataFrameQuery;

    List<Double> queryList = new ArrayList<>();
    String[] querySplit = stringQuery.split(delimiter);
    for (String s : querySplit) {
      queryList.add(Double.valueOf(s.trim()));
    }
    listQuery = ExplainerUtils.getSubLists(queryList, 1);
    dataFrameQuery = ExplainerUtils.dataframeFromList(listQuery, this.expParams.getDelimiter());

    // Continuous and categorical split
    DataFrame continuousFeatures = featuresDataFrame;
    DataFrame categoricalFeatures;
    String[] categoricalNames, continuousNames;
    int n = 0;
    categoricalNames = new String[categoricalList.size()];
    for (Integer col : categoricalList) {
      categoricalNames[n] = inputData.columns()[col];
      continuousFeatures = continuousFeatures.drop(categoricalNames[n++]);
    }
    continuousNames = continuousFeatures.columns();
    categoricalFeatures = featuresDataFrame.selectExpr(categoricalNames);

    // Discretization
    List<List<Double>> discretizedContinuous = new ArrayList<>();
    List<List<Double>> listDiscretized, listContinuous, queryDiscretized;
    listContinuous = ExplainerUtils.dataframeToList(continuousFeatures);
    if (this.expParams.isDiscretized() && (listContinuous.size() != 0)) {
      List<List<Double>> listPercentiles =
          ExplainerUtils.calculatePercentiles(listContinuous, percentileValues);
      discretizedContinuous = ExplainerUtils.discretize(listContinuous, listPercentiles);
      listDiscretized =
          ExplainerUtils.replaceContinuousSamples(featuresList, discretizedContinuous,
              categoricalList);
      List<List<Double>> queryContinuous =
          ExplainerUtils.constructListWithColumnNames(dataFrameQuery, continuousNames);
      queryDiscretized =
          ExplainerUtils.replaceContinuousSamples(listQuery,
              ExplainerUtils.discretize(queryContinuous, listPercentiles), categoricalList);
    } else {
      listDiscretized = ExplainerUtils.dataframeToList(categoricalFeatures); // listCategorical
      queryDiscretized =
          ExplainerUtils.constructListWithColumnNames(dataFrameQuery, categoricalNames); // queryCategorical
    }

    // Values and Probabilities
    List<List<Double>> listProbabilities = new ArrayList<List<Double>>();
    List<List<Double>> listValues = new ArrayList<List<Double>>();
    long numRows = inputData.count();
    for (List<Double> list : listDiscretized) {
      List<Double> distinctValues = new ArrayList<Double>();
      List<Double> probability = new ArrayList<Double>();
      for (Double d : list) {
        if (!distinctValues.contains(d)) {
          distinctValues.add(d);
          probability.add(Double.valueOf(Collections.frequency(list, d)) / numRows);
        }
      }
      listValues.add(distinctValues);
      listProbabilities.add(probability);
    }

    // Sampling Using Probabilities for Categorical Columns
    List<List<Double>> listSamples = new ArrayList<>();
    for (int i = 0; i < listProbabilities.size(); i++) {
      listSamples.add(ExplainerUtils.weightedSamplingWithReplacement(listValues.get(i),
          listProbabilities.get(i), numSamples));
    }

    // Binary column used in calculations
    List<List<Double>> listBinary;
    List<Double> binary = new ArrayList<>();
    for (int i = 0; i < listSamples.size(); i++) {
      for (int j = 0; j < numSamples; j++) {
        if ((j == 0) || (listSamples.get(i).get(j).equals(queryDiscretized.get(i).get(0)))) {
          binary.add(1.0);
        } else {
          binary.add(0.0);
        }
      }
    }
    listBinary = ExplainerUtils.getSubLists(binary, numSamples);

    // Finding mean and standard deviation
    List<Double> listMean = ExplainerUtils.findMean(featuresList);
    List<Double> listStdDev = ExplainerUtils.findStdDev(featuresList);

    // Random Sampling for Continuous Columns
    if (!(this.expParams.isDiscretized())) {
      List<List<Double>> continuousSamples = new ArrayList<>();
      for (int i = 0; i < featuresList.size(); i++) {
        continuousSamples.add(ExplainerUtils.randomSamplingFromNormal(listMean.get(i),
            listStdDev.get(i), numSamples));
      }
      listSamples =
          ExplainerUtils.replaceCategoricalSamples(continuousSamples, listSamples, categoricalList);
      listBinary =
          ExplainerUtils.replaceCategoricalSamples(continuousSamples, listBinary, categoricalList);
    }

    // Undiscretize if discretized
    List<List<Double>> undiscretizedSamples;
    if ((this.expParams.isDiscretized()) && (listContinuous.size() != 0)) {
      List<List<Double>> sampleLists =
          ExplainerUtils.constructListWithColumnNames(
              ExplainerUtils.dataframeFromList(listSamples, this.expParams.getDelimiter()),
              continuousNames);
      undiscretizedSamples =
          ExplainerUtils.replaceContinuousSamples(listSamples, ExplainerUtils.undiscretize(
              discretizedContinuous, listContinuous, sampleLists, percentileValues),
              categoricalList);
    } else {
      undiscretizedSamples = listSamples;
    }

    // Replacing the first row of the undiscretized samples with the query
    for (int i = 0; i < undiscretizedSamples.size(); i++) {
      undiscretizedSamples.get(i).set(0, listQuery.get(i).get(0));
    }

    // Altering means and standard deviations for categorical columns
    if ((this.expParams.isDiscretized()) && (listContinuous.size() != 0)) {
      for (int i = 0; i < listMean.size(); i++) {
        listMean.set(i, 0.0);
        listStdDev.set(i, 1.0);
      }
    } else {
      int l = 0;
      for (int i = 0; i < listMean.size(); i++) {
        if (categoricalList.size() != 0) {
          if (i == (categoricalList.get(l) - 1)) {
            listMean.set(i, 0.0);
            listStdDev.set(i, 1.0);
            if (l < (categoricalList.size() - 1)) {
              l++;
            }
          }
        }
      }
    }

    // Scaling
    List<List<Double>> listScaled = new ArrayList<>();
    List<Double> scaled;
    Double mean;
    Double std;
    int k = 0;
    for (List<Double> list : listBinary) {
      scaled = new ArrayList<>();
      mean = listMean.get(k);
      std = listStdDev.get(k);
      for (int i = 0; i < list.size(); i++) {
        scaled.add((list.get(i) - mean) / std);
      }
      listScaled.add(scaled);
      k++;
    }

    // Euclidean distance calculation
    List<Double> euclideanDistance = new ArrayList<>();
    DataFrame scaledDataFrame =
        ExplainerUtils.dataframeFromList(listScaled, this.expParams.getDelimiter());
    Row[] binaryRows = scaledDataFrame.collect();
    double sum;
    for (int i = 0; i < binaryRows.length; i++) {
      sum = 0;
      for (int j = 0; j < numColumns; j++) {
        sum +=
            Math.pow((Double.valueOf(binaryRows[0].get(j).toString()) - Double
                .valueOf(binaryRows[i].get(j).toString())), 2);
      }
      euclideanDistance.add(Math.sqrt(sum));
    }

    // Calculating Weights
    List<Double> listWeights = new ArrayList<>();
    double kernelWidth = Math.sqrt(numColumns) * 0.75;
    for (int i = 0; i < euclideanDistance.size(); i++) {
      listWeights.add(Math.sqrt(Math.exp(-(Math.pow(euclideanDistance.get(i), 2))
          / Math.pow((kernelWidth), 2))));
    }

    // Class Names
    List<String> classNames = new ArrayList<>();
    Row[] distinct = inputData.select(label).distinct().collect();
    for (Row name : distinct) {
      classNames.add(name.get(0).toString());
    }

    // Feature Names
    List<String> featureNames = new ArrayList<>();
    if (this.expParams.isColumnNameSpecified()) {
      featureNames.addAll(Arrays.asList(actualColNames));
      featureNames.remove(0);
    } else {
      for (int i = 0; i < numColumns; i++) {
        builder = new StringBuilder("F").append(i + 1);
        featureNames.add(builder.toString());
      }
    }

    List<String> names = new ArrayList<>();
    if ((this.expParams.isDiscretized()) && (listContinuous.size() != 0)) {
      List<List<Double>> listPercentiles =
          ExplainerUtils.calculatePercentiles(listContinuous, percentileValues);
      String name;
      for (int i = 0; i < continuousNames.length; i++) {
        name =
            featureNames
                .get(Integer.valueOf(continuousNames[i].replaceAll("[^0-9]", "").trim()) - 1);
        builder = new StringBuilder(name).append("<=").append(listPercentiles.get(i).get(0));
        names.add(builder.toString());
        for (int j = 0; j < (listPercentiles.get(i).size() - 1); j++) {
          builder =
              new StringBuilder().append(listPercentiles.get(i).get(j)).append("<").append(name)
                  .append("<=").append(listPercentiles.get(i).get(j + 1));
          names.add(builder.toString());
        }
        builder =
            new StringBuilder(name).append(">").append(
                listPercentiles.get(i).get((listPercentiles.get(i).size()) - 1));
        names.add(builder.toString());
      }
    } else {
      for (int i = 0; i < continuousNames.length; i++) {
        for (int j = 0; j < (percentileValues.length + 1); j++) {
          names.add(featureNames.get(Integer.valueOf(continuousNames[i].replaceAll("[^0-9]", "")
              .trim()) - 1));
        }
      }
    }

    List<List<String>> listNames = new ArrayList<List<String>>();
    for (int i = 0; i < names.size(); i += (percentileValues.length + 1)) {
      listNames.add(new ArrayList<String>(names.subList(i,
          Math.min(i + (percentileValues.length + 1), names.size()))));
    }

    // Feature Names Categorical
    List<String> features = new ArrayList<>(featureNames);
    int y = 0;
    for (int i = 0; i < numColumns; i++) {
      if (categoricalList.size() != 0) {
        if (i == (categoricalList.get(y) - 1)) {
          builder =
              new StringBuilder(features.get(i)).append("=").append(
                  listQuery.get(i).get(0).intValue());
          features.set(i, builder.toString());
          if (y < (categoricalList.size() - 1)) {
            y++;
          }
        }
      }
    }

    // Feature Names discretized
    List<String> transformedFeatureNames = new ArrayList<>();
    List<String> featureValues = new ArrayList<>();
    int u = 0;
    int v = 0;
    for (int i = 0; i < numColumns; i++) {
      if (categoricalList.size() != 0) {
        if (i == (categoricalList.get(v) - 1)) {
          transformedFeatureNames.add(features.get(i));
          featureValues.add("True");
          if (v < ((categoricalList.size()) - 1)) {
            v++;
          }
        } else {
          if ((this.expParams.isDiscretized()) && (listContinuous.size() != 0)) {
            transformedFeatureNames.add(listNames.get(u).get(
                queryDiscretized.get(i).get(0).intValue()));
          } else {
            transformedFeatureNames.add(listNames.get(u).get(0));
          }
          featureValues.add(listQuery.get(i).get(0).toString());
          if (u < (listNames.size() - 1)) {
            u++;
          }
        }
      } else {
        if (this.expParams.isDiscretized() && (listContinuous.size() != 0)) {
          transformedFeatureNames.add(listNames.get(i).get(
              queryDiscretized.get(i).get(0).intValue()));
        } else {
          transformedFeatureNames.add(listNames.get(i).get(0));
        }
        featureValues.add(listQuery.get(i).get(0).toString());
      }
    }

    // Class Probabilities
    List<List<Double>> predictProbability = new ArrayList<>();
    List<String> inverselist =
        ExplainerUtils.getAppendedList(undiscretizedSamples, this.expParams.getDelimiter());
    List<Double> queryProbability = new ArrayList<>();
    List<Double> probabilityValues;

    Map<String, Map<String, String>> probabilityMap =
        new LinkedHashMap<String, Map<String, String>>();
    Map<String, String> probMap;
    DecimalFormat twoDForm = new DecimalFormat("#.######");
    org.apache.spark.mllib.linalg.Vector featureVector;
    DecisionTreeModel[] trees;
    double[] featuresArr;
    String[] splitter;
    String prediction;
    for (int i = 0; i < inverselist.size(); i++) {
      probMap = new LinkedHashMap<>();
      splitter = inverselist.get(i).split(delimiter);
      featuresArr = new double[splitter.length];
      for (int j = 0; j < splitter.length; j++) {
        featuresArr[j] = Double.parseDouble(splitter[j].trim());
      }
      featureVector = Vectors.dense(featuresArr);
      trees = model.trees();
      for (DecisionTreeModel tree : trees) {
        prediction = String.valueOf(tree.predict(featureVector));
        if (probMap.containsKey(prediction)) {
          String value =
              twoDForm.format(Double.parseDouble(probMap.get(prediction)) + (1.00f / trees.length));
          probMap.put(String.valueOf(prediction), value);
        } else {
          probMap.put(String.valueOf(prediction),
              String.valueOf(Double.valueOf(twoDForm.format(1.00f / trees.length))));
        }
        for (int t = 0; t < classNames.size(); t++) {
          if (!probMap.keySet().contains(String.valueOf(Double.valueOf(classNames.get(t))))) {
            probMap.put(String.valueOf(Double.valueOf(classNames.get(t))), "0");
          }
        }
      }
      probabilityMap.put(String.valueOf(i), probMap);
    }

    for (int t = 0; t < classNames.size(); t++) {
      probabilityValues = new ArrayList<>();
      for (String s : probabilityMap.keySet()) {
        probabilityValues.add(Double.valueOf(probabilityMap.get(s).get(
            String.valueOf(Double.valueOf(classNames.get(t))))));
      }
      queryProbability.add(probabilityValues.get(0));
      predictProbability.add(probabilityValues);
    }

    // Weighted data and label
    List<List<Double>> wtdLabels =
        ExplainerUtils.dataWithSampleWeights(listWeights, predictProbability,
            this.expParams.getDelimiter());
    List<List<Double>> wtdData =
        ExplainerUtils
            .dataWithSampleWeights(listWeights, listBinary, this.expParams.getDelimiter());
    List<String> appendedData = new ArrayList<>();
    List<Double> labelNeeded = new ArrayList<>();
    List<Double> appendedDataPoints;
    String str1;
    for (int i = 0; i < (wtdData.size()); i++) {
      appendedDataPoints = new ArrayList<>();
      for (int j = 0; j < (wtdData.get(0).size() + 1); j++) {
        if (j == 0) {
          appendedDataPoints.add(wtdLabels.get(i).get((int) (labelToBeExplained)));
          labelNeeded.add(wtdLabels.get(i).get((int) (labelToBeExplained)));
        } else {
          appendedDataPoints.add(wtdData.get(i).get(j - 1));
        }
      }
      str1 = appendedDataPoints.toString();
      appendedData.add(str1.substring(str1.indexOf("[") + 1, str1.lastIndexOf("]")));
    }

    List<List<Double>> appendedColumnlists = new ArrayList<>();
    List<Double> appendedColumnDataPoints;
    String str2 = appendedData.toString();
    String[] strArray;
    strArray = str2.substring(str2.indexOf("[") + 1, str2.lastIndexOf("]")).split(",");
    for (int i = 0; i < (numColumns + 1); i++) {
      appendedColumnDataPoints = new ArrayList<>();
      for (int j = i; j < strArray.length; j = (j + numColumns + 1)) {
        appendedColumnDataPoints.add(Double.valueOf(strArray[j]));
      }
      appendedColumnlists.add(appendedColumnDataPoints);
    }

    List<List<Double>> appendedDataWithoutlabel = new ArrayList<>(appendedColumnlists);
    appendedDataWithoutlabel.remove(0);

    // Weights of each feature using ridge
    RidgeRegressionWithSGD obj;
    final RidgeRegressionModel ridgemodel;
    final Double[] weights;
    double[] featureWeights;
    Integer[] featureNos;
    if (this.expParams.isDiscretized()) {
      obj = new RidgeRegressionWithSGD();
    } else {
      obj = new RidgeRegressionWithSGD(stepSize, numIterations, 0.1, 1.0);
    }
    obj.setIntercept(true);
    JavaRDD<String> data = SparkUtils.getInstance().getJavaSparkContext().parallelize(appendedData);
    ridgemodel = obj.run(ExplainerUtils.convertRDDStringToLabeledPoint(data, ",").rdd());
    featureWeights = ridgemodel.weights().toArray();
    weights = new Double[featureWeights.length];
    for (int i = 0; i < featureWeights.length; i++) {
      weights[i] = featureWeights[i];
    }
    featureNos = new Integer[featureWeights.length];
    for (int i = 0; i < featureNos.length; i++) {
      featureNos[i] = i;
    }
    Arrays.sort(featureNos, (wts1, wts2) -> weights[wts2].compareTo(weights[wts1]));

    // Prediction probabilities
    Map<String, String> classMap = new LinkedHashMap<String, String>();
    for (int i = 0; i < classNames.size(); i++) {
      builder = new StringBuilder("'").append(classNames.get(i)).append("'");
      classMap.put(builder.toString(), String.valueOf(queryProbability.get(i)));
    }

    // Feature probabilities
    Map<String, String> featureMap = new LinkedHashMap<String, String>();
    for (int i = 0; i < numFeatures; i++) {
      builder =
          new StringBuilder("'").append(transformedFeatureNames.get(featureNos[i])).append("'");
      featureMap.put(builder.toString(), String.valueOf(featureWeights[featureNos[i]]));
    }

    // Feature values
    Map<String, String> featureQuery = new LinkedHashMap<String, String>();
    for (int i = 0; i < numFeatures; i++) {
      builder = new StringBuilder("'").append(features.get(featureNos[i])).append("'");
      featureQuery.put(builder.toString(), featureValues.get(featureNos[i]));
    }

    ExplainerResults expResult = new ExplainerResults();
    expResult.setPredictionProbabilities(classMap);
    expResult.setFeatureWeights(featureMap);
    expResult.setFeatureValues(featureQuery);
    return expResult;
  }

}
