# !/usr/bin/env python3
# -*- coding_ctrl: utf-8 -*-

# ------------------------------------- #
# Created by: Joao Pedro Peters Barbosa #
#       & Pedro Henrique Peters Barbosa #
#                                       #
# email: joao.peters@engenharia.ufjf.br #
#  or pedro.henrique@engenharia.ufjf.br #
# ------------------------------------- #


"""
Disciplina [210115] - Topicos Especiais em Otimizacao: Tecnicas Inteligentes

Desenvolvimento do programa referente ao segundo trabalho da disciplina

Modelo de Previsão de Velocidade de Ventos
Baseado em Redes Neurais Artificiais (RNA)

Prof.: Leonardo Willer de Oliveira
"""

# ----------------------- #
# -_-_- BIBLIOTECAS _-_-_ #
# ----------------------- #
import os
import subprocess
import sys

def requirements():
    pwd = os.getcwd()
    if os.path.isfile(pwd + "/.check.txt") is False:
        f = open(".check.txt", "w+")
        f.write("Bibliotecas instaladas:\n\n"
                "keras\nmatplotlib\nnumpy\nopenpyxl\npandas\nscikit-learn\ntensorflow\nxlrd\n\n"
                "by:phjp")
        f.close()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xlrd"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


