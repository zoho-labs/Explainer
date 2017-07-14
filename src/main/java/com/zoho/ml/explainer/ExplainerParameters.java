package com.zoho.ml.explainer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class ExplainerParameters implements Serializable {

  private static final long serialVersionUID = 1L;
  private int numSamples = 5000;
  private int numFeatures = 0;
  private int numIterations = 10;
  private double stepSize = 0.001;
  private String delimiter = ",";
  private String dataPath = null;
  private boolean discretize = true;
  private boolean columnNameSpecified = false;
  private int[] percentileValues = {25, 50, 75};
  private List<Integer> categoricalColumns = null;

  // Classifier Parameters
  private String impurity = "gini";
  private int maxDepth = 5;
  private int maxBins = 32;
  private int numClasses = 2;
  private int numTrees = 10;
  private int seed = 12345;
  private int minPartitions = 4;
  private String featureSubsetStrategy = "auto";

  public boolean isDiscretized() {
    return discretize;
  }

  public ExplainerParameters setDiscretize(boolean discretize) {
    this.discretize = discretize;
    return this;
  }

  public int getNumSamples() {
    return this.numSamples;
  }

  public ExplainerParameters setNumSamples(int numSamples) {
    this.numSamples = numSamples;
    return this;
  }

  public int[] getPercentileValues() {
    return this.percentileValues;
  }

  public ExplainerParameters setPercentileValues(int[] percentileValues) {
    this.percentileValues = percentileValues;
    return this;
  }

  public ExplainerParameters setNumberOfFeatures(int numFeatures) throws Exception {
    if (numFeatures < 0) {
      throw new Exception(this.getClass() + " : Number of Features cannot be Negative");
    }
    this.numFeatures = numFeatures;
    return this;
  }

  public int getNumberOfFeatures() {
    return this.numFeatures;
  }

  public ExplainerParameters setNumberOfIterations(int numIterations) throws Exception {
    if (numIterations < 0) {
      throw new Exception(this.getClass() + " : Number of Iterations cannot be Negative");
    }
    this.numIterations = numIterations;
    return this;
  }

  public int getNumberOfIterations() {
    return this.numIterations;
  }

  public ExplainerParameters setStepSize(double stepSize) throws Exception {
    if (stepSize < 0) {
      throw new Exception(this.getClass() + " : Step Size cannot be Negative");
    }
    this.stepSize = stepSize;
    return this;
  }

  public double getStepSize() {
    return this.stepSize;
  }

  public ExplainerParameters setDelimiter(String delimiter) {
    this.delimiter = delimiter;
    return this;
  }

  public String getDelimiter() {
    return this.delimiter;
  }

  public ExplainerParameters setCategoricalColumns(List<Integer> columns) {
    this.categoricalColumns = columns;
    return this;
  }

  public List<Integer> getCategoricalColumns() {
    return this.categoricalColumns;
  }

  public ExplainerParameters setColumnNameSpecified(boolean columnNameSpecified) {
    this.columnNameSpecified = columnNameSpecified;
    return this;
  }

  public boolean isColumnNameSpecified() {
    return this.columnNameSpecified;
  }

  public String getDataPath() {
    return this.dataPath;
  }

  public ExplainerParameters setDataPath(String dataPath) throws Exception {
    dataPath = dataPath != null ? dataPath.trim() : "";
    if (dataPath.equals("")) {
      throw new Exception(this.getClass() + " : DataPath cannot be null.");
    }
    this.dataPath = dataPath;
    return this;
  }

  public int getNumTrees() {
    return this.numTrees;
  }

  public ExplainerParameters setNumTrees(int numTrees) throws Exception {
    if (numTrees < 0) {
      throw new Exception(this.getClass()
          + " : Trees count is less than 0. Count should be greater than 0.");
    } else {
      this.numTrees = numTrees;
    }
    return this;
  }

  public int getSeed() {
    return this.seed;
  }

  public ExplainerParameters setSeed(int seed) throws Exception {
    this.seed = seed;
    return this;
  }

  public String getFeatureSubsetStrategy() {
    return this.featureSubsetStrategy;
  }

  public ExplainerParameters setFeatureSubsetStrategy(String featureSubsetStrategy)
      throws Exception {
    List<String> supportedStrategies = Arrays.asList("auto", "all", "sqrt", "log2", "onethird");
    if ((featureSubsetStrategy == null)
        || !supportedStrategies.contains(featureSubsetStrategy.toLowerCase())) {
      throw new Exception(
          this.getClass()
              + " : Feature Subset Strategy is either null or not supported. Supported strategies are all/sqrt/log2/onethird/auto(recommended).");
    } else {
      this.featureSubsetStrategy = featureSubsetStrategy.toLowerCase();
    }
    return this;
  }

  public String getImpurity() {
    return this.impurity;
  }

  public ExplainerParameters setImpurity(String impurity) throws Exception {
    List<String> supportedImpurities = Arrays.asList("entropy", "gini");
    if (impurity == null || !supportedImpurities.contains(impurity.toLowerCase())) {
      throw new Exception(
          this.getClass()
              + " : Impurity is either null or not supported. Supported Impurities gini(Recommended) or entropy.");
    } else {
      this.impurity = impurity.toLowerCase();
    }
    return this;
  }

  public int getMaxDepth() {
    return this.maxDepth;
  }

  public ExplainerParameters setMaxDepth(int maxDepth) throws Exception {
    if (maxDepth < 0 || maxDepth > 30) {
      throw new Exception(this.getClass() + " : Depth is not within the range. Depth range : 0-30.");
    } else {
      this.maxDepth = maxDepth;
    }
    return this;
  }

  public int getMaxBins() {
    return this.maxBins;
  }

  public ExplainerParameters setMaxBins(int maxBins) throws Exception {
    if (maxBins < 2) {
      throw new Exception(this.getClass()
          + " : Bins value is less than 2. The value should be >= 2.");
    } else {
      this.maxBins = maxBins;
    }
    return this;
  }

  public int getNumClasses() {
    return this.numClasses;
  }

  public ExplainerParameters setNumClasses(int numClasses) throws Exception {
    if (numClasses < 2) {
      throw new Exception(this.getClass()
          + " : Number of classes is less than 2. Value should be >= 2.");
    } else {
      this.numClasses = numClasses;
    }
    return this;
  }

  public int getMinPartitions() {

    return this.minPartitions;
  }

  public ExplainerParameters setMinPartitions(int minPartitions) throws Exception {
    if (minPartitions < 4) {
      throw new Exception(this.getClass()
          + " : Minimum number of Partitions is less than 4. Value should be >= 4");
    } else {
      this.minPartitions = minPartitions;
    }
    return this;
  }

}
