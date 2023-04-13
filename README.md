# Wind Velocity Prediction Model via Neural Networks

## Pedro Henrique Peters Barbosa & João Pedro Peters Barbosa 

contact: [pedro.peters@engenharia.ufjf.br](pedro.peters@engenharia.ufjf.br), [joao.peters@engenharia.ufjf.br](joao.peters@engenharia.ufjf.br) 

---

This repo contains the simulation results regarding the implementation of neural networks for wind velocity prediction. 

The Neural Network model adopted in this project is of the LSTM type (*Long Short-Term Memory Network*). This topology is applied in a wide range of problems that perform time series forecasting.

The simple neural network model developed by the duo proposes the use of only one hidden layer of the LSTM type to predict historical data, using *n* inputs to predict *1* output value.

In all, the group applied the Neural Network in order to carry out two analyses:
1. "**n**" amount of **past hours** to forecast **a specific hour ahead**;
2. "**n**" amount of **past days** to forecast the **same respective hour one day ahead**.


The main files present in this repo are:

1) [windneural.py](./main/windneural.py): 
	- Manipulates the historical data considering the analyzes to be carried out;
	- Determines the parameters of the Neural Network model developed;
	- Carries out graphical analyzes of the results obtained.

> The historical data used in the model can be found [here.](./main/).

2) [assistant.py](./main/assistant.py)
	- Verifies and automatically installs the Python libraries used for the correct functioning of the simulation.

```sh
keras

matplotlib

numpy

openpyxl

pandas

scikit-learn

tensorflow

xlrd
```


The examples folder contains different Python codes with neural networks and machine learning implementations.


---
#### References
[Jason Brownlee - Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras]

[Jason Brownlee - Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras ]: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
