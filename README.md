# ML-in-equity-prediction

This repository contains code, data, and documentation about my thesis titled "Empirical Asset Pricing via Deep Learning" for the University of British Columbia's ECON 493 class. The original work was done on googlecollab servers, which I then converted to a basic python script along with a jupyter notebook for illustration purposes. I also provide my thesis paper, and the related presentation

Using LSTM networks, I predict one-month-ahead stock returns. My data contains fundamental and technical variables for companies over 40 years, macroeconomic variables for the same period and their interactions, totaling more than 900 variables. I am not uploading the dataset because of copyright concerns but the SAS code that creates the dataset is provided with proper citations. Also, the sources of data are stated in my thesis paper, which is provided in this directory.

The inspiration for this thesis came from: "Empirical Asset Pricing via Machine Learning" by Gu etal. located at: "https://dachxiu.chicagobooth.edu/download/ML.pdf"

Novelties of this research paper is:
1) Applies a novel method for estimating the significance of coefficients in a neural network model which I cross-check with my prior expectations
2) Uses recurrent neural networks for predicting one-month ahead stock returns
3) Provide statistical data about different neural network configurations using cross-validation
4) Provide evidence for the self-regularization effect of LSTM networks
