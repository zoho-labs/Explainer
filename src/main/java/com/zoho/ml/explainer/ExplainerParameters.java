package com.zoho.ml.explainer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

@SuppressWarnings("serial")
public class ExplainerParameters implements Serializable {

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
  private Integer maxDepth = 5;
  private Integer maxBins = 32;
  private Integer numClasses = 2;
  private Integer numTrees = 10;
  private Integer seed = 12345;
  private Integer minPartitions = 4;
  private String featureSubsetStrategy = "auto";

  public boolean isDiscretized() {
    return discretize;
  }

  public ExplainerParameters setDiscretize(boolean discretize) {
    this.discretize = discretize;
    return this;
  }

  public int getNumSamples() {
    return numSamples;
  }

  public ExplainerParameters setNumSamples(int numSamples) {
    this.numSamples = numSamples;
    return this;
  }

  public int[] getPercentileValues() {
    return percentileValues;
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
    return numFeatures;
  }

  public ExplainerParameters setNumberOfIterations(int numIterations) throws Exception {
    if (numIterations < 0) {
      throw new Exception(this.getClass() + " : Number of Iterations cannot be Negative");
    }
    this.numIterations = numIterations;
    return this;
  }

  public int getNumberOfIterations() {
    return numIterations;
  }

  public ExplainerParameters setStepSize(double stepSize) throws Exception {
    if (stepSize < 0) {
      throw new Exception(this.getClass() + " : Step Size cannot be Negative");
    }
    this.stepSize = stepSize;
    return this;
  }

  public double getStepSize() {
    return stepSize;
  }

  public ExplainerParameters setDelimiter(String delimiter) {
    this.delimiter = delimiter;
    return this;
  }

  public String getDelimiter() {
    return delimiter;
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
    return columnNameSpecified;
  }

  public String getDataPath() {
    return dataPath;
  }

  public ExplainerParameters setDataPath(String dataPath) throws Exception {
    dataPath = dataPath != null ? dataPath.trim() : "";
    if (dataPath.equals("")) {
      throw new Exception(this.getClass() + " : DataPath cannot be null.");
    }
    this.dataPath = dataPath;
    return this;
  }

  public Integer getNumTrees() {
    return numTrees;
  }

  public ExplainerParameters setNumTrees(Integer numTrees) throws Exception {
    if (numTrees == null || numTrees < 0) {
      throw new Exception(this.getClass()
          + " : Trees count is either null or less than 0. Count should be greater than 0.");
    } else {
      this.numTrees = numTrees;
    }
    return this;
  }

  public Integer getSeed() {
    return seed;
  }

  public ExplainerParameters setSeed(Integer seed) throws Exception {
    if (seed == null) {
      throw new Exception(this.getClass() + " : Seed value cannot be null.");
    } else {
      this.seed = seed;
    }
    return this;
  }

  public String getFeatureSubsetStrategy() {
    return featureSubsetStrategy;
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
    return impurity;
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

  public Integer getMaxDepth() {

    return maxDepth;
  }

  public ExplainerParameters setMaxDepth(Integer maxDepth) throws Exception {
    if (maxDepth == null || maxDepth < 0 || maxDepth > 30) {
      throw new Exception(this.getClass()
          + " : Depth is either null or not within the range. Depth range : 0-30.");
    } else {
      this.maxDepth = maxDepth;
    }
    return this;
  }

  public Integer getMaxBins() {
    return maxBins;
  }

  public ExplainerParameters setMaxBins(Integer maxBins) throws Exception {
    if (maxBins == null || maxBins < 2) {
      throw new Exception(this.getClass()
          + " : Bins value is either null or less than 2. The value should be >= 2.");
    } else {
      this.maxBins = maxBins;
    }
    return this;
  }

  public Integer getNumClasses() {
    return numClasses;
  }

  public ExplainerParameters setNumClasses(Integer numClasses) throws Exception {
    if (numClasses == null || numClasses < 2) {
      throw new Exception(this.getClass()
          + " : Number of classes is either null or less than 2. Value should be >= 2.");
    } else {
      this.numClasses = numClasses;
    }
    return this;
  }

  public Integer getMinPartitions() {

    return minPartitions;
  }

  public ExplainerParameters setMinPartitions(Integer minPartitions) throws Exception {
    if (minPartitions == null || minPartitions < 4) {
      throw new Exception(this.getClass()
          + " : Minimum number of Partitions is either null or less than 4. Value should be >= 4");
    } else {
      this.minPartitions = minPartitions;
    }
    return this;
  }

}
