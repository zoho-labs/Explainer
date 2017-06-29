package com.zoho.ml.explainer.client;

import com.zoho.ml.explainer.Explainer;
import com.zoho.ml.explainer.ExplainerParameters;
import com.zoho.ml.explainer.ExplainerResults;



public class Client {

  public static void main(String[] args) throws Exception {
    ExplainerParameters expParams =
        new ExplainerParameters().setDataPath("/data/Iris.csv").setColumnNameSpecified(true)
            .setNumClasses(3);
    Explainer exp = new Explainer(expParams);
    ExplainerResults expResult = exp.explain("5.4,3.4,1.7,0.2");
    System.out.println(expResult.toString());
    System.out.println(expResult.getPredictionProbabilities());
    System.out.println(expResult.getFeatureWeights());
    System.out.println(expResult.getFeatureValues());
  }

}
