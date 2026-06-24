-------------------------------------------------------------------------------------------------------

**Notebook 1 (exploration.ipynb):** The starting point of the project. It ingests the raw NFL time-series data and lays the foundation for subsequent modeling stages.

**Details:**

* Assesses the quality, integrity, and dimensionality of the raw dataset.
* Isolates the target variable (**Yards**) to prevent data leakage.
* Performs an initial cleaning process by removing irrelevant columns (such as weather information and redundant textual data).
* Standardizes key data types, including height conversion, age calculation, and game clock fraction computation.

---

**Notebook 2 (EDA.ipynb):** The goal is to understand correlations, distributions, and player/team behavior in order to guide feature engineering decisions.

**Details:**

* Loads and analyzes the preprocessed dataset stored in Parquet format.
* Examines the distributions of numerical and categorical variables associated with rushing plays.
* Investigates the impact of physical attributes (weight, height, and age) and positional characteristics on yardage gained.
* Develops geospatial visualizations and tactical field maps showing the positioning of all 22 players at the exact moment of the handoff.

---

**Notebook 3 (feature_eng.ipynb):** Feature creation and transformation based on the cleaned data, focusing on the mathematical modeling of spatial and dynamic interactions between players.

**Details:**

* Loads the previously consolidated dataset and initiates a structured analytical workflow.
* Develops pairwise interaction geometry between the rusher and the 11 defenders by calculating Euclidean distances, axial components (X/Y), decomposed angles, and relative kinematics (closing speed).
* Implements Time-to-Collision (TTC) calculations and projected minimum distance metrics to estimate when and how closely a defender will reach the rusher.
* Constructs features describing interactions between the rusher and the 10 offensive blockers to identify open running lanes and blocking support.
* Applies data type optimizations (float32/int32) to reduce memory consumption and prepares the final wide-format table merged with the target variable (**Yards**) for model training.

---

**Notebook 4 (baselines.ipynb):** Initial experimentation and benchmark establishment. The objective is to evaluate different model architectures and define robust validation metrics to guide project development.

**Details:**

* Implements a game-level K-Fold cross-validation strategy based on **GameId** to prevent temporal data leakage.
* Defines and implements the official competition metric, **CRPS (Continuous Ranked Probability Score)**.
* Builds and trains traditional baseline and gradient boosting models (such as LightGBM and XGBoost), mapping predictions into a 199-step cumulative distribution (from -99 to 99 yards).
* Develops and evaluates a **Multi-Layer Perceptron (MLP)** designed to directly predict the probability distribution of yardage gains.

---

**Notebook 5 (graph_models.ipynb):** Graph Neural Network (GNN) modeling. The goal is to represent each play as a complete player graph in order to capture complex interactions and spatial relationships in an unstructured manner.

**Details:**

* Develops and evaluates graph-based architectures using PyTorch Geometric, experimenting with **Graph Attention Network (GAT)** layers.
* Constructs and injects node-level features (physical characteristics, positioning, and player kinematics) as well as global play-level context features.
* Implements validation strategies aligned with previous notebooks to ensure direct performance comparisons.
* Conducts multiple experiments varying network depth and model capacity to optimize the probabilistic loss metric (**CRPS**).

---

**Notebook 6 (mixture_of_specialists.ipynb):** Mixture Density Networks & Calibration. The objective is to implement a neural architecture capable of modeling continuous probabilistic outputs and applying calibration techniques to improve tail-distribution predictions.

**Details:**

* Implements a **Mixture Density Network (MDN)** in PyTorch to model yardage uncertainty through a mixture of Gaussian distributions.
* Develops a custom data generator with support for data augmentation via spatial field mirroring (inverting the Y-axis of plays).
* Applies post-processing **Beta calibration** to out-of-fold (OOF) predictions in order to correct probabilistic biases and directly optimize CRPS.
* Performs tail-focused diagnostics to assess model accuracy in extreme scenarios (large losses or long gains).
* Exports the final trained calibrators as a serialized **.pkl** file for production deployment.

---

**Notebook 7 (mlp_bagging.ipynb):** Robustness and stability optimization of the best-performing strategy. The objective is to create an ensemble of MLP networks with varying initializations and data subsets to reduce variance and generate the final production artifacts.

**Details:**

* Implements a bagging strategy by training multiple independent MLP models with different random seeds, dropout rates, and feature subsets.
* Incorporates training-time data augmentation through field mirroring (Y-axis transformations).
* Combines individual predictions using a weighted average based on each model's validation performance, reducing overall CRPS.
* Consolidates and exports all critical pipeline artifacts, including network weights (.pt files), the normalization scaler, categorical mappings, and feature definitions into a single serialized file (**nfl_mlp_artifacts.pkl**).

---

**Notebook 8 (mlp_fine_tuning.ipynb):** The objective is to perform a grid search to optimize the Beta calibrator parameters, ensuring maximum probabilistic accuracy for ensemble predictions.

**Details:**

* Loads the predictions and cumulative distribution functions (CDFs) generated by the MLP ensemble.
* Implements a custom grid search to evaluate different regularization levels (**C parameter**) across three distinct distribution regions: the body, left tail, and right tail.
* Assesses the impact of each parameter combination directly on CRPS using out-of-fold (OOF) validation.
* Analyzes calibration robustness, confirming convergence and stability regardless of the tested regularization range.
* Exports the optimized and fully parameterized calibrators into a final serialized file (**nfl_beta_calibrators_tuned.pkl**) ready for production deployment.

---

**Notebook 9 (final_moe.ipynb):** The goal is to implement a Mixture of Experts (**MoE**) architecture coupled with a hybrid tails head and apply statistical calibration to predict NFL yardage distribution.

**Details:**

* Develops a custom head (**SplicedWeibullParetoHead**) that divides the play into three tactical regions: left tail (Reflected Weibull), body (1-yard discrete histogram), and right tail (Generalized Pareto).
* Implements dynamic threshold logic that learns the exact transition limits between the body and the heavy tails of the model adaptively on a play-by-play basis.
* Applies a post-processing pipeline using non-linear **Beta Calibration** to align predicted probabilities with actual frequencies observed on the field.
* Mitigates severe confidence distortions generated by direct optimization of the **CRPS (Continuous Ranked Probability Score)** metric.
* Consolidates final comparative statistical reports to validate the exact calibration gain and predictive performance on test data.
  
-------------------------------------------------------------------------------------------------------

**Notebook 1 (exploration.ipynb):** Ponto de partida do projeto. Ele recebe os dados brutos de séries temporais da NFL e prepara o terreno para as modelagens futuras.

**Detalhamento:**

* Avalia a qualidade, integridade e dimensionalidade da base bruta.
* Isola a variável alvo (**Yards**) para garantir que não haja data leakage.
* Realiza uma limpeza inicial eliminando colunas irrelevantes (como clima e dados textuais redundantes).
* Padroniza tipos de dados fundamentais (conversão de alturas, cálculo de idades e frações de tempo do relógio da partida).

---

**Notebook 2 (EDA.ipynb):** O objetivo é compreender as correlações, distribuições e o comportamento dos jogadores e das equipas para orientar a engenharia de atributos (*feature engineering*).

**Detalhamento:**

* Carrega e analisa a base de dados pré-processada no formato Parquet.
* Estuda as distribuições das variáveis numéricas e categóricas associadas às jogadas de corrida.
* Explora o impacto das características físicas (peso, altura e idade) e de posicionamento no ganho de jardas.
* Desenvolve visualizações geoespaciais e mapas táticos do posicionamento dos 22 jogadores em campo no momento exato do *handoff*.

---

**Notebook 3 (feature_eng.ipynb):** Criação e transformação de variáveis a partir dos dados limpos, focando na modelagem matemática das interações espaciais e dinâmicas entre os jogadores.

**Detalhamento:**

* Carrega os dados previamente unificados e inicia o processo estruturado em etapas analíticas.
* Desenvolve a geometria de interações par a par entre o corredor (*Rusher*) e os 11 defensores, calculando distâncias euclidianas, componentes axiais (X/Y), ângulos decompostos e cinemática relativa (velocidade de aproximação).
* Implementa o cálculo do **Time-to-Collision (TTC)** e da menor distância projetada para prever em quanto tempo e quão perto um defensor alcançará o corredor.
* Constrói atributos para as interações entre o corredor e os 10 bloqueadores de ataque para identificar caminhos livres e suporte à corrida.
* Aplica otimizações de tipos de dados (**float32/int32**) para reduzir o uso de memória e prepara a tabela final (*wide format*) cruzada com o alvo (**Yards**) para o treino dos modelos.

---

**Notebook 4 (baselines.ipynb):** Experimentação inicial e estabelecimento de referências de desempenho. O objetivo é testar diferentes arquiteturas de modelos e definir métricas de validação robustas para orientar a evolução do projeto.

**Detalhamento:**

* Implementa uma estratégia de validação cruzada (**K-Fold**) baseada nas partidas (**GameId**) para evitar vazamento de dados temporais.
* Define e codifica a métrica oficial de avaliação do problema: o **CRPS (Continuous Ranked Probability Score)**.
* Constrói e treina modelos de referência tradicionais e de *Gradient Boosting* (como **LightGBM** e **XGBoost**), mapeando as previsões para uma distribuição acumulada de 199 passos (de -99 a 99 jardas).
* Desenvolve e avalia uma rede neuronal do tipo **MLP (Multi-Layer Perceptron)** estruturada para prever diretamente a distribuição probabilística de ganho de jardas.

---

**Notebook 5 (graph_models.ipynb):** Modelagem utilizando **GNNs (Graph Neural Networks)**. O objetivo é tratar cada jogada como um grafo completo de jogadores para capturar interações complexas e relacionamentos espaciais de forma não estruturada.

**Detalhamento:**

* Desenvolve e testa arquiteturas de redes baseadas em grafos utilizando **PyTorch Geometric**, experimentando com camadas do tipo **GAT (Graph Attention Networks)**.
* Constrói e injeta atributos específicos para cada nó (características físicas, posicionamento e cinemática de cada jogador) e atributos globais para o contexto da jogada.
* Implementa estratégias de validação consistentes com os notebooks anteriores (alinhando os *splits* de dados) para permitir uma comparação direta de desempenho.
* Realiza múltiplos experimentos alterando a profundidade das camadas e a capacidade do modelo para otimizar a métrica de perda probabilística (**CRPS**).

---

**Notebook 6 (mixture_of_specialists.ipynb):** **Mixture Density Networks & Calibration**. O objetivo é implementar uma arquitetura neural capaz de modelar saídas probabilísticas contínuas e aplicar técnicas de calibração para otimizar as previsões na cauda da distribuição.

**Detalhamento:**

* Implementa uma **MDN (Mixture Density Network)** em **PyTorch** para modelar a incerteza do ganho de jardas através de uma mistura de distribuições gaussianas.
* Desenvolve um gerador de dados customizado com suporte a *data augmentation* via espelhamento espacial do campo (invertendo o eixo Y das jogadas).
* Aplica calibração **Beta** em pós-processamento nas previsões **OOF (Out-Of-Fold)** para corrigir desvios probabilísticos e otimizar diretamente a métrica **CRPS**.
* Realiza diagnósticos focados nas caudas da distribuição para avaliar a precisão do modelo em cenários extremos (grandes perdas ou avanços longos).
* Exporta os calibradores finais treinados num ficheiro serializado (**.pkl**) para utilização em produção.

---

**Notebook 7 (mlp_bagging.ipynb):** Otimização de robustez e estabilidade da melhor estratégia experimentada. O objetivo é criar um *ensemble* baseado em múltiplas redes neuronais **MLP** com variações de inicialização e subconjuntos de dados para minimizar a variância e gerar os artefatos finais para produção.

**Detalhamento:**

* Implementa uma estratégia de **Bagging** treinando múltiplos modelos **MLP** independentes sob diferentes sementes (*seeds*), taxas de *dropout* e partições de atributos (*feature subsets*).
* Incorpora técnicas de aumento de dados em tempo de treino por meio do espelhamento do campo (ajuste das colunas no eixo Y).
* Combina as previsões individuais através de uma média ponderada calculada a partir do desempenho de cada rede para reduzir o erro geral (**CRPS**).
* Consolida e exporta todos os artefatos estruturais cruciais do *pipeline*, incluindo os pesos das redes em arquivos **.pt**, o objeto de normalização (*scaler*), os mapeamentos de categorias e as colunas utilizadas num único arquivo serializado (**nfl_mlp_artifacts.pkl**).

---

**Notebook 8 (mlp_fine_tuning.ipynb):** O objetivo é realizar um *grid search* para otimizar os parâmetros do calibrador **Beta**, garantindo a máxima precisão probabilística nas previsões do *ensemble*.

**Detalhamento:**

* Carrega as previsões e as funções de distribuição acumulada (**CDFs**) geradas pelo *ensemble* de **MLPs**.
* Implementa uma busca em grelha customizada para testar diferentes níveis de regularização (**parâmetro C**) aplicados a três regiões distintas da distribuição: corpo, cauda esquerda e cauda direita.
* Avalia o impacto de cada combinação diretamente na métrica **CRPS** através de validação **OOF (Out-Of-Fold)**.
* Analisa a robustez da calibração, constatando a convergência e estabilidade dos resultados independentemente do intervalo de regularização testado.
* Exporta os calibradores ótimos e parametrizados num arquivo serializado final (**nfl_beta_calibrators_tuned.pkl**) pronto para o ambiente de produção.

---

**Notebook 9 (final_moe.ipynb):** O objetivo é implementar uma arquitetura de Mistura de Especialistas (**MoE**) acoplada a uma cabeça de caudas híbridas e aplicar calibração estatística para prever a distribuição de jardas da NFL.

**Detalhamento:**

* Desenvolve uma cabeça customizada (**SplicedWeibullParetoHead**) que divide a jogada em três regiões táticas: cauda esquerda (Weibull Refletida), corpo (histograma discreto de 1 jarda) e cauda direita (Pareto Generalizada).
* Implementa uma lógica de thresholds dinâmicos que aprende os limites exatos de transição entre o corpo e as caudas pesadas do modelo de forma adaptativa por jogada.
* Aplica um pipeline de pós-processamento utilizando **Calibração Beta** não-linear para alinhar as probabilidades preditas com as frequências reais observadas em campo.
* Mitiga distorções de confiança severas geradas pela otimização direta da métrica **CRPS (Continuous Ranked Probability Score)**.
* Consolida relatórios estatísticos comparativos finais para validar o ganho exato de calibração e a performance preditiva nos dados de teste.

