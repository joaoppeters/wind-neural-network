# !/usr/bin/env python3
# -*- coding_ctrl: utf-8 -*-

# ------------------------------------- #
# Created by: JOaO PEDRO PETERS BARBOSA #
#       & PEDRO HENRIQUE PETERS BARBOSA #
#                                       #
# email: joao.peters@engenharia.ufjf.br #
#  or pedro.henrique@engenharia.ufjf.br #
# ------------------------------------- #


"""
Disciplina [210115] - Topicos Especiais em Otimizacao: Tecnicas Inteligentes

Desenvolvimento do programa referente ao segundo trabalho da disciplina

Modelo de Previsão de Velocidade de Ventos Baseado em Redes Neurais Artificiais
Aplicacao Basica do Modelo LSTM: Long Short-Term Memory

Prof.: Leonardo Willer de Oliveira
"""

# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------- BIBLIOTECAS ---------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
from InstallAssistant_PHJP import requirements
requirements()

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------- #
# MENSAGEM SOBRE LEITURA DE BIBLIOTECAS #
print("BIBLIOTECAS IMPORTADAS COM SUCESSO!\n")


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------- DEFINIcaO BaSICA DE PARaMETROS: MATPLOTLIB, NUMPY, PANDAS & TENSORFLOW ------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
# DEFINE PARaMETROS DO MATPLOTLIB #
mpl.rcParams["figure.figsize"] = (9, 6)
mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8

# -------------------------- #
# DEFINE PARaMETROS DO NUMPY #
np.set_printoptions(precision=3, suppress=False)

# --------------------------- #
# DEFINE PARaMETROS DO PANDAS #
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.3f}".format)


# ------------------------------------------- #
# DEFINE PARaMETROS DE MENSAGEM DO TENSORFLOW #
tf.get_logger().setLevel("ERROR")

# -------------------------------------- #
# MENSAGEM SOBRE DEFINIcaO DE PARAMETROS #
print("PARAMETROS DEFINIDOS COM SUCESSO!\n")


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------- FUNcaO DE PLOTAGEM DOS DADOS INFORMADOS -------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def plot_data(df, x, y, hue, xlabel, ylabel, flag=None):
    """
    PLOTAGEM DE DADOS SELECIONADOS
    -----

    :param df: dataframe -- informacao do dataframe selecionado
    :param x: array -- informacao dos valores para eixo das abscissas
    :param y: array -- informacao dos valores para eixo das ordenadas
    :param hue: str -- informacao para classificacao da plotagem
    :param xlabel: str -- informacao do texto para rotular o eixo das abscissas
    :param ylabel: str -- informacao do texto para rotular o eixo das ordenadas
    :param flag: int -- informacao para tipo de plotagem a ser realizada

    :return: -x-
    """

    fig, ax = plt.subplots(1)
    plt.ylim([-3, 12])
    days = ["dia {}".format(d) for d in range(1, 32)]
    hours = np.arange(0, 24)
    if flag is None:
        for n in range(0, len(y)):
            if (hue == "hora"):
                n -= 1
            plt.plot(x, df[y[n]], label=hue + " {}".format(n + 1))

        if (hue == "dia"):
            plt.xticks(x, hours)
        else:
            plt.xticks(x, days, rotation="vertical")
        ax.legend(labelspacing=0, loc=6, bbox_to_anchor=(1, 0.5),
                  fontsize=12, frameon=False)
    else:
        if (flag == 1):
            plt.plot(x, df.mean().values, label="Valor Médio por {}".format(hue.capitalize()))
            high = df.mean().values + df.std().values
            low = df.mean().values - df.std().values
            plt.plot(x, high, color="#7f7f7f")
            plt.plot(x, low, color="#7f7f7f")
            plt.fill_between(x, high, low, alpha=0.25, facecolor="#7f7f7f", label="Desvio Padrão da Média")
            if (hue == "hora"):
                plt.xticks(x, hours)
            else:
                plt.xticks(x, days, rotation="vertical")
            ax.legend(fontsize=12, frameon=False)

        if (flag == 2):
            plt.plot(x, df.values)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ FUNcaO PARA CRIAcaO DE DATASET ------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
def create_dataset(dataset, nInput, nOut, future):
    """
    SELECIONA E ORGANIZA EM VARIaVEIS OS DADOS A SEREM UTILIZADOS NO PROPOSTO PROBLEMA
    nInputs PARA nOut EM UM PASSO future a FRENTE
    -----

    :param dataset: array -- dados do vetor a serem desagregados e organizados conforme nInput
    :param nInput: int -- quantidade de parametros de entrada
    :param nOut: int -- quantidade de parametros de saida
    :param future: int -- previsao a n passos a frente

    :return dataX: array -- utilizado como input no modelo ML criado
    :return dataY: array -- utilizado como output no modelo ML criado
    """

    dataX, dataY = [], []
    for i in range(0, dataset.shape[0] - nInput - 1):
        a = dataset[i:(i + nInput), 0]
        dataX.append(a)
        dataY.append(dataset[i + nInput + future - 1, 0])

    return np.array(dataX), np.array(dataY)


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- FUNcaO DE CRIAcaO DO MODELO -------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def build_model(check_input, check_loss, check_optimizer, check_activation, check_neurons):
    """
    CRIA E COMPILA UM MODELO LSTM SIMPLES
    -----

    :param check_input: int -- quantidade de parametros de entrada
    :param check_loss: str -- tipo de erro a ser analisado
    :param check_optimizer: str -- otimizador selecionado para o modelo
    :param check_activation: str -- funcao de ativacao para o modelo
    :param check_neurons: int -- numero de neuronios (2/3*nInput + 1/3*nOutput)

    :return model: tf -- modelo de redes neurais criado
    """

    # ---------------------------- #
    # MAIS SIMPLES MODELO tf.keras #
    model = Sequential()

    # ---------------------------------------------------------------------------------- #
    # OPcaO PARA DEFINIcaO DOS PARaMETROS DO OTIMIZADOR                                  #
    # AS INFORMAcoES DO OTIMIZADOR FORAM ENCONTRADAS NO                                  #
    # SEGUINTE LINK: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam #
    tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999,
                             epsilon=1e-07, amsgrad=False, name="adam")

    # ---------------------------------------- #
    # DESCRIcaO DA TOPOLOGIA DO MODELO ADOTADO #
    model.add(LSTM(units=check_neurons, input_shape=(1, check_input)))
    model.add(Dense(1, activation=check_activation))

    # ------------------------------------------------------------------------------- #
    # CONFIGURAcaO DO TREINAMENTO PARA MINIMIZAcaO DO ERRO MeDIO QUADRaTICO DO MODELO #
    check_metrics = tf.keras.metrics.RootMeanSquaredError()

    # ----------------------------------- #
    # COMPILAcaO DO MODELO DE REDE NEURAL #
    model.compile(
        loss=check_loss,
        metrics=[check_metrics],
        optimizer=check_optimizer)

    # ----------------------------------------------- #
    # MENSAGEM SOBRE CRIAcaO DO MODELO DE REDE NEURAL #
    model.summary()
    print("MODELO DE REDE NEURAL LSTM CRIADO COM SUCESSO!\n")

    return model


# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- FUNcaO DE TREINAMENTO DO MODELO ------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
def train_model(check_model, check_X, check_Y, check_epochs, check_batch_size, check_verbose):
    """
    TREINAMENTO DO MODELO POR MEIO DA ALIMENTAcaO DOS DADOS SELECIONADOS
    -----

    :param check_model: tf -- modelo de redes neurais criado anteriormente
    :param check_X: array -- dados de input devidamente organizados
    :param check_Y: array -- dados de output devidamente organizados
    :param check_epochs: int -- quantidade de epocas para treinamento do modelo
    :param check_batch_size: int -- quantidade de valores passados a cada epoca de treinamento
    :param check_verbose: int, str -- estrategia de treinamento selecionada

    :return trained_weight: array -- informacoes sobre dos pesos calculados pela rede neural
    :return trained_bias: array -- informacao sobre o peso base calculado pela rede neural
    :return epochs: int -- quantidade total de epocas simuladas
    :return hist: df -- informacoes sobre o processo de treinamento
    """

    # ------------------------------------------------ #
    # ALIMENTAcaO DO MODELO COM DADOS PARA TREINAMENTO #
    history = check_model.fit(x=check_X, y=check_Y,
                              batch_size=check_batch_size,
                              epochs=check_epochs,
                              verbose=check_verbose)

    # ----------------------------------------------- #
    # ARMAZENAMENTO DAS INFORMAcoES DE PESO CALCULADO #
    trained_weight = check_model.get_weights()[0]
    trained_bias = check_model.get_weights()[1]

    # -------------------------------------------------- #
    # ARMAZENAMENTO DAS INFORMAcoES DE ePOCAS REALIZADAS #
    epochs = history.epoch

    # ----------------------------------------------- #
    # ARMAZENAMENTO DAS INFORMAcoES DE ERRO POR ePOCA #
    hist = pd.DataFrame(history.history)

    # ------------------------------------ #
    # MENSAGEM SOBRE TREINAMENTO DO MODELO #
    print("TREINAMENTO DO MODELO REALIZADO COM SUCESSO!\n")

    return trained_weight, trained_bias, epochs, hist


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- FUNcaO PARA PLOTAGEM DA VARIAcaO DO ERRO POR ePOCA -------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def plot_loss_curve(epochs, error, ylabel, actv_func):
    """
    PLOTAGEM DE CURVA loss x epochs
    -----

    :param epochs: int -- numero de epocas realizadas no treinamento do modelo
    :param error: array -- erro calculado por epoca
    :param ylabel: str -- tipo de erro a ser analisado
    :param actv_func: str -- funcao de ativacao para treinamento do modelo de rede neural

    :return: -x-
    """

    plt.figure()
    plt.xlabel("Epoca")
    plt.ylabel(ylabel)

    plt.plot(epochs, error["root_mean_squared_error"], label="Loss - " + actv_func.upper())
    plt.legend(fontsize=12, frameon=False)
    # plt.show()

    return


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PROCESSAMENTOS DOS DADOS --------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# LEITURA DOS DADOS #
df = pd.read_excel("grupo 3-1.xlsx", index_col=0).reset_index(drop=False, inplace=False)
print("DADOS LIDOS COM SUCESSO!\n")

# ------------------------------------------------------------------------ #
# PLOT VERTICAL POR PARTES: VELOCIDADE DO VENTO POR HORA, DE TODOS OS DIAS #
wind_df = df.copy().rename_axis("dia", axis="columns")
plot_data(wind_df, x=np.arange(1, 25, 1), y=wind_df.columns.to_list()[1:], hue="dia",
          xlabel="Hora do Dia", ylabel="Velocidade do Vento")

# ------------------------------------------------------------------------- #
# PLOT VERTICAL MeDIO: VELOCIDADE MeDIA DO VENTO POR HORA, DE TODOS OS DIAS #
TMH_wind_df = df.T.drop(index="hora").mean()
plot_data(df.T.drop(index="hora"), x=np.arange(1, 25, 1), y=TMH_wind_df.index.to_list(), hue="hora",
          xlabel="Hora do Dia", ylabel="Velocidade do Vento", flag=1)

# -------------------------------------------------------------------------- #
# PLOT HORIZONTAL POR PARTES: VELOCIDADE DO VENTO POR DIA, DE TODAS AS HORAS #
T_wind_df = df.T.drop(index="hora")
T_wind_df = T_wind_df.rename_axis("dia", axis="index").reset_index(drop=False, inplace=False)
plot_data(T_wind_df, x=np.arange(1, 32, 1), y=T_wind_df.columns.to_list()[1:], hue="hora",
          xlabel="Dia do Mês", ylabel="Velocidade do Vento")

# --------------------------------------------------------------------------- #
# PLOT HORIZONTAL MeDIO: VELOCIDADE MeDIA DO VENTO POR DIA, DE TODAS AS HORAS #
TMD_wind_df = df.copy().drop(columns="hora").mean()
plot_data(df.drop(columns="hora"), x=np.arange(1, 32, 1), y=TMD_wind_df.index.to_list(), hue="dia",
          xlabel="Dia do Mês", ylabel="Velocidade do Vento", flag=1)


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------- PRIMEIRA ANaLISE ------------------------------------------------- #
# --------------------------- horas passadas sendo utilizadas para prever horas a frente ----------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# DADO UTILIZADO #
month_df = wind_df.drop(columns="hora").T.stack()
stack_month_df = month_df.values.reshape(-1, 1)

# -------------------------------- #
# PLOT CURVA COMPLETA: 0h1d/23h31d #
plt.figure()
plt.plot(stack_month_df, label="Curva Original")
plt.legend(loc="upper left", fontsize=12, frameon=False)
plt.ylim([-3, 12])
plt.xlabel("Horas Total do Mês")
plt.ylabel("Velocidade do Vento")

# ----------------------------------------------- #
# NORMALIZAcaO DOS DADOS: ESCALA ENTRE ZERO E hUM #
scaler = MinMaxScaler(feature_range=(0, 1))
stack_month_df = scaler.fit_transform(stack_month_df)

# -------------------------------------------------------------------------------- #
# 70% TREINAMENTO (0h1d/23h22d) & 30% TESTE (0h23d/23h31d): AJUSTADOS MANUALMENTE! #
size = 528  # quantidade de dados para treinamento
train = stack_month_df[:size, :]  # variavel para armazenar os valores de treinamento
teste = stack_month_df[size:, :]  # variavel para armazenar os valores de teste

# ------------------------------------------- #
# CRIAcaO DO DATASET & PASSAGEM DE PARaMETROS #
nInput = 5  # numero de dados de entrada
nOut = 1  # numero de dados de saida
future = 1  # previsao a quantos passos a frente
trainXone, trainYone = create_dataset(train, nInput, nOut, future)  # determinacao das variaveis de treinamento
testeXone, testeYone = create_dataset(teste, nInput, nOut, future)  # determinacao das variaveis de teste

# --------------------------------- #
# RESHAPE DOS DATASETS DESAGREGADOS #
trainXone = np.reshape(trainXone, (trainXone.shape[0], 1, trainXone.shape[-1]))
testeXone = np.reshape(testeXone, (testeXone.shape[0], 1, testeXone.shape[-1]))

# ----------------------------------------------------------------------- #
# CRIAcaO DO MODELO: PASSAGEM DA FUNcaO DE ATIVAcaO, ERRO A SER ANALISADO #
# OTIMIZADOR SELECIONADO, NuMERO TOTAL DE NEURoNIOS NA CAMADA OCULTA      #
# NuMERO TOTAL DE ePOCAS, BATCH_SIZE & ERROS ANALISADOS                   #
activations = ["relu", "sigmoid", "tanh"]  # funcoes de ativacao
loss = "mean_squared_error"  # erro a ser analisado no treinamento
optimizer = "adam"  # otimizador selecionado
nNeurons = int((nInput * 0.67) + (nOut * 0.33))  # quantidade total de neuronios: 2/3*nInput + 1/3*nOut
nEpochs = 100  # quantidade total de epocas
nBatch_size = 4  # tamanho do batch analisado por epoca
errors = ["RMSE", "MAPE", "uTHEIL"]

# ------------------------------------------------------------------------------------------- #
# LOOP PARA VERIFICAcaO DOS RESULTADO DE TREINAMENTO EM FUNcaO DIFERENTES FUNcoES DE ATIVAcaO #
for actv_func in activations:
    # -------------------- #
    # CONSTRUcaO DO MODELO #
    model = build_model(check_input=nInput, check_loss=loss, check_optimizer=optimizer,
                        check_activation=actv_func, check_neurons=nNeurons)

    # -------------------------------------------- #
    # TREINAMENTO DO MODELO ePOCAS, BATCH P/ ePOCA #
    weight, bias, epochs, hist = train_model(check_model=model, check_X=trainXone,
                                             check_Y=trainYone, check_epochs=nEpochs,
                                             check_batch_size=nBatch_size, check_verbose="auto")
    # ----------------- #
    # PLOT loss x epoca #
    plot_loss_curve(epochs, hist, "Root Mean Squared Error", actv_func)

    # ----------------------------------- #
    # RESULTADOS DO TREINAMENTO DO MODELO #
    trainPredicted = model.predict(trainXone)
    testePredicted = model.predict(testeXone)

    # ------------------------- #
    # DESNORMALIZAcaO DOS DADOS #
    trainPredicted = scaler.inverse_transform(trainPredicted)
    trainExpected = scaler.inverse_transform([trainYone])
    testePredicted = scaler.inverse_transform(testePredicted)
    testeExpected = scaler.inverse_transform([testeYone])

    # --------------------------- #
    # LOOP PARA CaLCULO DOS ERROS #
    for e in errors:
        if (e == "RMSE"):
            rmse = np.sqrt(np.sum((trainExpected - trainPredicted.T) ** 2) / nEpochs)
            print("RMSE: {}".format(rmse))

        elif (e == "MAPE"):
            mape = np.sum(np.abs(trainExpected - trainPredicted.T)) / nEpochs * 100
            print("MAPE: {:.3f}%".format(mape))

        elif (e == "uTHEIL"):
            theil = np.sqrt(np.sum((trainExpected - trainPredicted.T) ** 2)) / np.sqrt(np.sum((trainExpected[:, 1:] - trainExpected[:, :-1]) ** 2))
            print("uTHEIL: {}".format(theil))

    # -------------------------------------------- #
    # SHIFT DOS DADOS DE TREINAMENTO PARA PLOTAGEM #
    trainPredictedPlot = np.empty_like(stack_month_df)
    trainPredictedPlot[:, :] = np.nan
    trainPredictedPlot[nInput:len(trainPredicted) + nInput, :] = trainPredicted

    # -------------------------------------- #
    # SHIFT DOS DADOS DE TESTE PARA PLOTAGEM #
    testePredictedPlot = np.empty_like(stack_month_df)
    testePredictedPlot[:, :] = np.nan
    testePredictedPlot[len(trainPredicted) + (nInput * 2) + 1:len(stack_month_df) - 1, :] = testePredicted

    # -------------------------------- #
    # PLOT CURVA PRINCIPAL E PREVISoeS #
    plot_data(month_df, x=np.arange(0, 31 * 24, 1), y=month_df.values, hue=None,
              xlabel="Horas Total do Mês", ylabel="Velocidade do Vento", flag=2)
    plt.text(529 * 1.05, 10, "Base de Dados\nde Teste", fontsize=12, fontweight="demibold", color="green")
    plt.text(529 * 0.65, 10, "Base de Dados\nde Treinamento", fontsize=12, fontweight="demibold", color="#ff7f0e")
    plt.plot(trainPredictedPlot)
    plt.plot(testePredictedPlot)
    plt.axvline(529, linestyle="--", color="red")
    plt.legend(["Curva Original", "Previsão Base de Treinamento", "Previsão Base de Teste"],
               loc="upper left", fontsize=12, frameon=False)
    # plt.show()


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------- SEGUNDA ANaLISE ------------------------------------------------- #
# ---------- hora de determinado números de dias sendo utilizada para prever esta mesma hora UM dia à frente --------- #
# -------------------------------------------------------------------------------------------------------------------- #
# MANIPULAcaO DOS DADOS #
hours_df = df.drop(columns="hora").T

# ------------------------------------------------------ #
# 70% TREINAMENTO (dia1-dia22) & 30% TESTE (dia23-dia31) #
size = 23

# -------------------------- #
# OUTPUT TREINAMENTO E TESTE #
trainYout = {}  # armazena os valores de saida do treinamento
testeYout = {}  # armazena os valores de saida do teste

# ----------------------------------------------------------------------- #
# CRIAcaO DO MODELO: PASSAGEM DA FUNcaO DE ATIVAcaO, ERRO A SER ANALISADO #
# OTIMIZADOR SELECIONADO, NuMERO TOTAL DE NEURoNIOS NA CAMADA OCULTA      #
# NuMERO TOTAL DE ePOCAS, BATCH_SIZE & ERROS ANALISADOS                   #
activations = "tanh"  # funcoes de ativacao
loss = "mean_squared_error"  # erro a ser analisado no treinamento
optimizer = "adam"  # otimizador selecionado
nNeurons = int(nInput * 0.67 + nOut * 0.33)  # quantidade total de neuronios: 2/3*nInput + 1/3*nOut
nEpochs = 50  # quantidade total de epocas
nBatch_size = 1  # tamanho do batch analisado por epoca
errors = ["RMSE", "MAPE", "uTHEIL"]
nInput = 1  # numero de dados de entrada
nOut = 1  # numero de dados de saida
future = 1  # previsao a quantos passos a frente

# # -------------------- #
# # CONSTRUcaO DO MODELO #
# model = build_model(check_input=nInput, check_loss=loss, check_optimizer=optimizer,
#                     check_activation=actv_func, check_neurons=nNeurons)

# ------------------------- #
# LOOP PARA SEGUNDA ANaLISE #
hours_day = 24  # quantidade total de horas no dia
for hc in range(0, hours_day):  # hc: hours_count - determina o vetor do df a ser lido
    # ---------------------------------------- #
    # ARMAZENA A MESMA HORA DE DIFERENTES DIAS #
    stack_hours_df = hours_df[hc].values.reshape(-1, 1)

    # ----------------------------------------------- #
    # NORMALIZAcaO DOS DADOS: ESCALA ENTRE ZERO E hUM #
    stack_hours_df = scaler.fit_transform(stack_hours_df)

    # ------------------------------------------------------------ #
    # ARMAZENAMENTO DOS VALORES: 70% - 30%: AJUSTADOS MANUALMENTE! #
    train = stack_hours_df[:size, :]
    teste = np.append(stack_hours_df[size - 1:, :], stack_hours_df[0, :]).reshape(-1, 1)

    # ------------------------------------------- #
    # CRIAcaO DO DATASET & PASSAGEM DE PARaMETROS #
    trainXtwo, trainYtwo = create_dataset(train, nInput, nOut, future)  # determinacao das variaveis de treinamento
    testeXtwo, testeYtwo = create_dataset(teste, nInput, nOut, future)  # determinacao das variaveis de teste

    # --------------------------------- #
    # RESHAPE DOS DATASETS DESAGREGADOS #
    trainXtwo = np.reshape(trainXtwo, (trainXtwo.shape[0], 1, trainXtwo.shape[-1]))
    testeXtwo = np.reshape(testeXtwo, (testeXtwo.shape[0], 1, testeXtwo.shape[-1]))

    # -------------------- #
    # CONSTRUcaO DO MODELO #
    model = build_model(check_input=nInput, check_loss=loss, check_optimizer=optimizer,
                        check_activation=actv_func, check_neurons=nNeurons)

    # -------------------------------------------- #
    # TREINAMENTO DO MODELO ePOCAS, BATCH P/ ePOCA #
    weight, bias, epochs, hist = train_model(check_model=model, check_X=trainXtwo,
                                             check_Y=trainYtwo, check_epochs=nEpochs,
                                             check_batch_size=nBatch_size, check_verbose="auto")

    # ----------------------------------- #
    # RESULTADOS DO TREINAMENTO DO MODELO #
    trainPredicted = model.predict(trainXtwo)
    testePredicted = model.predict(testeXtwo)

    # ------------------------- #
    # DESNORMALIZAcaO DOS DADOS #
    trainPredicted = scaler.inverse_transform(trainPredicted)
    trainExpected = scaler.inverse_transform([trainYtwo])
    testePredicted = scaler.inverse_transform(testePredicted)
    testeExpected = scaler.inverse_transform([testeYtwo])

    # --------------------------- #
    # LOOP PARA CaLCULO DOS ERROS #
    for e in errors:
        if (e == "RMSE"):
            rmse = np.sqrt(np.sum((trainExpected - trainPredicted.T) ** 2) / nEpochs)
            print("RMSE: {}".format(rmse))

        elif (e == "MAPE"):
            mape = np.sum(np.abs(trainExpected - trainPredicted.T)) / nEpochs * 100
            print("MAPE: {:.3f}%".format(mape))

        elif (e == "uTHEIL"):
            theil = np.sqrt(np.sum((trainExpected - trainPredicted.T) ** 2)) / np.sqrt(
                np.sum((trainExpected[:, 1:] - trainExpected[:, :-1]) ** 2))
            print("uTHEIL: {}".format(theil))

    # -------------------------------------------------------- #
    # ARMAZENAMENTO DOS VALORES PREVISTOS: BASE DE TREINAMENTO #
    trainYout[hc] = np.append(np.nan, trainPredicted)

    # -------------------------------------------------- #
    # ARMAZENAMENTO DOS VALORES PREVISTOS: BASE DE TESTE #
    testeYout[hc] = np.append(np.nan, testePredicted)

# --------------------------- #
# MANIPULAcaO FINAL DOS DADOS #
trainFinal = pd.DataFrame.from_dict(trainYout, orient="index")
trainFinal = trainFinal.T.stack(dropna=False)
testeFinal = pd.DataFrame.from_dict(testeYout, orient="index")
testeFinal = testeFinal.T.stack(dropna=False)

# -------------------------------- #
# PLOT CURVA PRINCIPAL E PREVISoeS #
plot_data(month_df, x=np.arange(0, 31 * 24, 1), y=month_df.values, hue=None,
          xlabel="Horas Total do Mês", ylabel="Velocidade do Vento", flag=2)
plt.text(529 * 1.05, 10, "Base de Dados\nde Teste", fontsize=12, fontweight="demibold", color="green")
plt.text(529 * 0.65, 10, "Base de Dados\nde Treinamento", fontsize=12, fontweight="demibold", color="#ff7f0e")
plt.plot(np.arange(0, 528), trainFinal)
plt.plot(np.arange(528, 744), testeFinal)
plt.axvline(529, linestyle="--", color="red")
plt.legend(["Curva Original", "Previsão Base de Treinamento", "Previsão Base de Teste"],
           loc="upper left", fontsize=12, frameon=False)
plt.show()
