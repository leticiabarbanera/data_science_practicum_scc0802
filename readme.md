Notebook 1 (exploration.ipynb): Ponto de partida do projeto. Ele recebe os dados brutos de séries temporais da NFL e prepara o terreno para as modelagens futuras.

Detalhamento: * Avalia a qualidade, integridade e dimensionalidade da base bruta.
Isola a variável alvo (Yards) para garantir que não haja data leakage. Realiza uma limpeza inicial eliminando colunas irrelevantes (como clima e dados textuais redundantes). Padroniza tipos de dados fundamentais (conversão de alturas, cálculo de idades e frações de tempo do relógio da partida).


Notebook 2 (EDA.ipynb): O objetivo é compreender as correlações, distribuições e o comportamento dos jogadores e das equipas para orientar a engenharia de atributos (feature engineering).

Detalhamento: * Carrega e analisa a base de dados pré-processada no formato Parquet.
Estuda as distribuições das variáveis numéricas e categóricas associadas às jogadas de corrida. Explora o impacto das características físicas (peso, altura e idade) e de posicionamento no ganho de jardas. Desenvolve visualizações geoespaciais e mapas táticos do posicionamento dos 22 jogadores em campo no momento exato do passe (handoff).


Notebook 3 (feature_eng.ipynb): Criação e transformação de variáveis a partir dos dados limpos, focando na modelagem matemática das interações espaciais e dinâmicas entre os jogadores.

Detalhamento: * Carrega os dados previamente unificados e inicia o processo estruturado em etapas analíticas.
Desenvolve a geometria de interações par a par entre o corredor (Rusher) e os 11 defensores, calculando distâncias euclidianas, componentes axiais (X/Y), ângulos decompostos e cinemática relativa (velocidade de aproximação). Implementa o cálculo do Time-to-Collision (TTC) e a menor distância projetada para prever em quanto tempo e quão perto um defensor alcançará o corredor. Constrói atributos para as interações entre o corredor e os 10 bloqueadores de ataque para identificar caminhos livres e suporte à corrida. Aplica otimizações de tipos de dados (float32/int32) para reduzir o uso de memória e prepara a tabela final (wide format) cruzada com o alvo (Yards) para o treino dos modelos.


Notebook 4 (baselines.ipynb): Experimentação inicial e estabelecimento de referências de desempenho. O objetivo é testar diferentes arquiteturas de modelos e definir métricas de validação robustas para orientar a evolução do projeto.

Detalhamento: * Implementa uma estratégia de validação cruzada (K-Fold) baseada nas partidas (GameId) para evitar vazamento de dados temporais.
Define e codifica a métrica oficial de avaliação do problema: o CRPS (Continuous Ranked Probability Score). Constrói e treina modelos de referência tradicionais e de Gradient Boosting (como LightGBM e XGBoost) mapeando as previsões para uma distribuição acumulada de 199 passos (de -99 a 99 jardas). Desenvolve e avalia uma rede neuronal do tipo MLP (Multi-Layer Perceptron) estruturada para prever diretamente a distribuição probabilística de ganho de jardas.


Notebook 5 (graph_models.ipynb): Modelagem utilizando GNNs. O objetivo é tratar cada jogada como um grafo completo de jogadores para capturar interações complexas e relacionamentos espaciais de forma não estruturada.

Detalhamento: * Desenvolve e testa arquiteturas de redes baseadas em grafos utilizando PyTorch Geometric, experimentando com camadas do tipo GAT (Graph Attention Networks).
Constrói e injeta atributos específicos para cada nó (características físicas, posicionamento e cinemática de cada jogador) e atributos globais para o contexto da jogada. Implementa estratégias de validação consistentes com os notebooks anteriores (alinhando os splits de dados) para permitir uma comparação direta de desempenho. Realiza múltiplos experimentos alterando a profundidade das camadas e a capacidade do modelo para otimizar a métrica de perda probabilística (CRPS).


Notebook 6 (mixture_of_especialists.ipynb): Mixture Density Networks & Calibration. O objetivo é implementar uma arquitetura neural capaz de modelar saídas probabilísticas contínuas e aplicar técnicas de calibração para otimizar as previsões na cauda da distribuição.

Detalhamento: * Implementa uma rede MDN (Mixture Density Network) em PyTorch para modelar a incerteza do ganho de jardas através de uma mistura de distribuições gaussianas.
Desenvolve um gerador de dados customizado com suporte a data augmentation via espelhamento espacial do campo (invertendo o eixo Y das jogadas). Aplica calibração Beta pós-processamento nas previsões OOF (Out-Of-Fold) para corrigir desvios probabilísticos e otimizar diretamente a métrica CRPS. Realiza diagnósticos focados nas caudas da distribuição para avaliar a precisão do modelo em cenários extremos (grandes perdas ou avanços longos). Exporta os calibradores finais treinados num ficheiro serializado (.pkl) para utilização em produção.


Notebook 7 (mlp_bagging.ipynb): Otimização de robustez e estabilidade da melhor estratégia experimentada. O objetivo é criar um ensemble baseado em múltiplas redes neuronais MLP com variações de inicialização e subconjuntos de dados para minimizar a variância e gerar os artefatos finais para produção.

Detalhamento: * Implementa uma estratégia de Bagging treinando múltiplos modelos MLP independentes sob diferentes sementes (seeds), taxas de dropout e partições de atributos (feature subsets).
Incorpora técnicas de aumento de dados em tempo de treino por meio do espelhamento do campo (ajuste das colunas no eixo Y). Combina as previsões individuais através de uma média ponderada calculada a partir do desempenho de cada rede para reduzir o erro geral (CRPS). Consolida e exporta todos os artefatos estruturais cruciais do pipeline, incluindo os pesos das redes em arquivos .pt, o objeto de normalização (scaler), os mapeamentos de categorias e as colunas utilizadas num único arquivo serializado (nfl_mlp_artifacts.pkl).


Notebook 8 (mlp_fine_tunning): O objetivo é realizar um grid search para otimizar os parâmetros do calibrador Beta, garantindo a máxima precisão probabilística nas previsões do ensemble.

Detalhamento: * Carrega as previsões e as funções de distribuição acumulada (CDFs) geradas pelo ensemble de MLPs.
Implementa uma busca em grelha customizada para testar diferentes níveis de regularização (parâmetro C) aplicados a três regiões distintas da distribuição: corpo, cauda esquerda e cauda direita. Avalia o impacto de cada combinação diretamente na métrica CRPS através de validação Out-Of-Fold (OOF). Analisa a robustez da calibração, constatando a convergência e estabilidade dos resultados independentemente do intervalo de regularização testado. Exporta os calibradores ótimos e parametrizados num arquivo serializado final (nfl_beta_calibrators_tuned.pkl) pronto para o ambiente de produção.

Notebook 9 (final_moe.ipynb): Mistura de Especialistas (MoE) com Caudas Híbridas Paramétricas e Calibração. 
Detalhamento: * O objetivo é implementar uma arquitetura estatística avançada que divide a predição de jardas em três regiões distintas (Weibull refletida para perdas, histograma para o centro e Pareto Generalizada para explosões) utilizando thresholds dinâmicos por jogada. O modelo aplica um pipeline de pós-processamento com Calibração Beta não-linear para alinhar as probabilidades com as frequências reais de campo, mitigando distorções de confiança e otimizando a métrica probabilística CRPS de forma consistente.
