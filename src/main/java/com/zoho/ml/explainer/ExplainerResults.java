package com.zoho.ml.explainer;

import java.util.Map;

public class ExplainerResults {

  private Map<String, String> predictionProbabilities;
  private Map<String, String> featureWeights;
  private Map<String, String> featureValues;

  public Map<String, String> getPredictionProbabilities() {
    return predictionProbabilities;
  }

  public void setPredictionProbabilities(Map<String, String> predictionProbabilities) {
    this.predictionProbabilities = predictionProbabilities;
  }

  public Map<String, String> getFeatureWeights() {
    return featureWeights;
  }

  public void setFeatureWeights(Map<String, String> featureWeights) {
    this.featureWeights = featureWeights;
  }

  public Map<String, String> getFeatureValues() {
    return featureValues;
  }

  public void setFeatureValues(Map<String, String> featureValues) {
    this.featureValues = featureValues;
  }

  @Override
  public String toString() {
    StringBuilder builder =
        new StringBuilder("PREDICTION PROBABILITIES" + " : " + predictionProbabilities + "\n");
    builder.append("FEATURE WEIGHTS" + " : " + featureWeights + "\n");
    builder.append("FEATURE VALUES" + " : " + featureValues);
    return builder.toString();
  }
}
