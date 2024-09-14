# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:26:17 2024

@author: orion
"""

#Importamos o modelo
from Models import Signal_and_Hierachical_Noise_2 as SHN2

'''Aqui definimos as listas com os trÊs tipos de dados:'''
dados_off = [29417,29668,30541,28392,28627,27366,27393]

dados_on = [30518,31605,31491,31849,32176,31496,31904,31619,30359,
            31118,30882,31396,29075,28053,29340]

dados_cut = [6313,6312,6325,5706,6060,5827,5840,6228,6021,6157,6488,
             6344,6105,6362,6214,6050,6245,6051,6289,6054,6048,6200]

'''Defimos os parametros de simulação (número de amostras e walkers)'''
samples = 24000
nwalkers = 260

'''Chama e inicializa a classe'''
test = SHN2.Signal_and_Hierachical_Noise_2(dados_off,dados_on,dados_cut,
                                           samples,nwalkers)

#%% CUIDADO: RODAR A SIMULAÇÃO APENAS PELA PRIMEIRA VEZ, SÓ PARA GERAR A CADEIA

'''Aqui rodamos a simulação e salvamos o resultado em um aquivo do tipo h5py'''
#chain = test.run(save=True,filename='Analise_25_08_24.h5')

#%%  RODAR PARA LER O ARQUIVO SALV0

'''Aqui roda os arquivos h5py salvos contendo a cadeia simulada'''
test.read_saved_chain('Analise_25_08_24.h5')

#%% Vamos ler a cadeia inicialmente e verificar o traço, para fazer escolha burn-in

'''função lê a cadeia e salva em modo numpy e Arviz(recebe o ponto de burn-in)'''
test.get_chain()

'''Gera o traço de todos os paremtros(recebe a quantidade de subfiguras 
nos graficos de traço de parametros de ruido para melhor adequar a forma)'''
test.trace_plot()

#%% Ler a cadeia novamente agora aplicando o burn-in

'''Agora lê e salva a cadeia com o corte de burn-in'''
test.get_chain(4000)

#%% Funções de graficos e tabelas

#Gráfico Traços
test.trace_plot()

#Graficos Posteriori de parametros de interesse
test.arviz_posterior_interest_plot()

#Graficos Posteriori de parametros de ruído variavel
test.posterior_graph_noise_params()

#Tabela com as medidas resumo
test.arviz_summary_stats()

#Gráfico Corner da Posteriori
test.corner_interest_param()

#Gráficos preditivos a posteriori dos 3 tipos de dados
test.pred_graph()

