# Explainer

One liner : A machine learning model explainer that works on top of Apache Spark!

This project is inspired from the python version hosted here : https://github.com/marcotcr/lime , which is based on a paper mentioned here : https://arxiv.org/abs/1602.04938

Today ML/AI is being used in mission critical applications. However, it is still difficult for a human being to trust a black-boxy ML algorithm. Wouldn’t it be cool if an algorithm could explain why it had predicted a particular result and strengthen it’s voice? Well, that is exactly what this project has achieved to do.

We at ZOHOCorp, heavily use Apache Spark for our Machine Learning activities, and since this explainer should work very close to the actual ML engine, we thought of rewriting the explainer that makes the best use of Apache Spark APIs and also resides closer to the actual ML engine. We are using a forked version of this in production now.

This project would be very useful for any ML practitioner using Apache Spark. Contributions welcome!
