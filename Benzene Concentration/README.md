# Benzene Concentration - Artificial Neural Networks (ANN)

Artificial  Neural  Network  (ANN)  is  an  advanced  machine  learning  technique  inspired  from  biological neural  networks.  The  technique  is  capable  of  taking  several  inputs  to  predict  the  outcome through a series of layers and user-defined   activation functions. This report summarizes the results obtained using an ANN model developed to predict the Benzene concertation in the air. This  outcome  is  influenced  by  11  predictors; Date,  Time,  Day, &  concentrations  of  Tin  Oxide,  Titania,  Tungsten  Oxide  (NOx),  Tungsten Oxide  (NO2),  Indium  Oxide,  &  the  effects  of  Temperature  (T),  Relative  Humidity  (RH),  and  Absolute  Humidity  (AH).     Preliminary  analysis  based on correlation coefficients was performed to understand the general behavior of the dataset. The  dataset  was  then  normalized  using  the  ‘zscore’  function  and  was  then  used  throughout  the  analysis. The dataset had was altered such that the Date was represented as number of days and time was expressed as the portion of the day lapsed, while others left as is. Using this dataset, a linear  regression  model  using  the  ‘fitlm’  function  was  obtained  in  addition  to  a  regression  tree  analysis. These gave a better perspective of the high impact predictors and the errors including the R2 error and the Mean-Squared Error (MSE). Go forth, a ANN model was developed with all the predictors and then the plots were constructed using just one predictor at a time, while keeping others at a constant value. The constant values were either the mean, minimum or maximum value. Furthermore, another ANN model was developed with just two high impact predictors and finally there was sensitivity analysis performed by changes in the number of neurons, activation functions and hidden layers.  


## Reference

This was from a graduate level class at Penn State, taught by Dr. Jeremy Gernand  


