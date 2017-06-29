package com.zoho.ml.explainer;

import java.util.LinkedHashMap;

public class ExplainerResults {

  private LinkedHashMap<String, String> predictionProbabilities;
  private LinkedHashMap<String, String> featureWeights;
  private LinkedHashMap<String, String> featureValues;

  public LinkedHashMap<String, String> getPredictionProbabilities() {
    return predictionProbabilities;
  }

  public void setPredictionProbabilities(LinkedHashMap<String, String> predictionProbabilities) {
    this.predictionProbabilities = predictionProbabilities;
  }

  public LinkedHashMap<String, String> getFeatureWeights() {
    return featureWeights;
  }

  public void setFeatureWeights(LinkedHashMap<String, String> featureWeights) {
    this.featureWeights = featureWeights;
  }

  public LinkedHashMap<String, String> getFeatureValues() {
    return featureValues;
  }

  public void setFeatureValues(LinkedHashMap<String, String> featureValues) {
    this.featureValues = featureValues;
  }

  @Override
  public String toString() {
    StringBuffer sb =
        new StringBuffer("PREDICTION PROBABILITIES" + " : " + predictionProbabilities + "\n");
    sb.append("FEATURE WEIGHTS" + " : " + featureWeights + "\n");
    sb.append("FEATURE VALUES" + " : " + featureValues);
    return sb.toString();
  }
}
