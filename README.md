# Sistema de Reconhecimento de Objetos em Tempo Real com CNN

Este é um projeto ponta a ponta (*end-to-end*) de Visão Computacional que coleta dados, treina uma Rede Neural Convolucional (CNN) customizada e realiza a inferência de objetos em tempo real utilizando a webcam. 

O modelo foi configurado por padrão para identificar três classes de objetos:
* **Classe 0:** Chapéu
* **Classe 1:** Facão
* **Classe 2:** Bota

---

## 🚀 Funcionalidades e Destaques Técnicos

* **Coleta de Dados Customizada:** Script automatizado para capturar e balancear frames diretamente da webcam.
* **Pré-processamento Avançado:** Conversão para tons de cinza, redução espacial para 32x32 pixels e **Equalização de Histograma** para neutralizar variações bruscas de iluminação e sombras.
* **Arquitetura Leve (CNN):** Modelo convolucional otimizado com Keras/TensorFlow para rodar em tempo real mesmo em CPUs convencionais.
* **Suavização Temporal por Média Móvel:** Sistema inteligente que armazena as predições dos últimos 8 frames para eliminar o efeito de oscilação (*flickering*) na tela.
* **Filtro de Confiança Defensivo:** Exibe o comando "Aproxime um objeto" caso a certeza do modelo seja inferior a 55%, evitando falsos positivos com cenários vazios.

---

## 📁 Estrutura do Projeto

```text
├── Imagens/               # Pasta gerada automaticamente com as fotos coletadas
│   ├── 0/                 # Fotos do Chapéu
│   ├── 1/                 # Fotos do Facão
│   └── 2/                 # Fotos da Bota
├── capturar_imagens.py    # Script para coleta e criação do dataset
├── treinar.py             # Script para pré-processamento e treino da CNN
├── webcam.py              # Script para inferência e HUD em tempo real
├── modelo.h5              # Arquivo do modelo treinado (gerado após o treino)
└── README.md              # Documentação do projeto
```
---
## 🛠️ Pré-requisitos e Instalação: 
Antes de começar, certifique-se de ter o Python 3.8+ instalado. Instale as dependências necessárias utilizando o gerenciador de pacotes pip:

```text
pip install opencv-python tensorflow scikit-learn matplotlib numpy
```
---

## 💻 Como Executar o Projeto
O pipeline do projeto é dividido em 3 etapas sequenciais:

1. Coleta de Imagens (capturar_imagens.py)
Abra o arquivo capturar_imagens.py e altere a variável CLASSE para o objeto que deseja registrar ("0", "1" ou "2"). Execute o script:

```text
python capturar_imagens.py
```

Como funciona: O script aguardará 3 segundos para você se posicionar e capturará automaticamente 50 fotos com intervalo de 100ms entre elas. Mexa o objeto devagar em vários ângulos para enriquecer o dataset. Repita o processo para as 3 classes.

2. Treinamento do Modelo (treinar.py)
Com as pastas de imagens devidamente preenchidas, execute o script de treinamento:

```
python treinar.py
```
Como funciona: O script lerá as fotos, aplicará o tratamento digital (Grayscale + Equalização + Resizing), dividirá o dataset de forma estratificada (80% treino / 20% teste) e treinará a CNN por 15 épocas com lotes de tamanho 8. Ao final, exibirá os gráficos de acurácia/perda e salvará o arquivo modelo.h5.

3. Inferência em Tempo Real (webcam.py)
Para ver o sistema funcionando ao vivo através da sua webcam, execute:

```
python webcam.py
``` 
Controles: Mostre os objetos cadastrados na tela. Uma barra de progresso visual indicará o nível de confiança da rede. Pressione a tecla q para fechar a janela de transmissão.

---

## 🧠 Arquitetura da Rede Neural (CNN)
O modelo foi construído utilizando a API Sequential do Keras com a seguinte topologia:

* Camada Convolucional 1 (Conv2D): 32 filtros (3x3), Ativação ReLU. Entrada: (32, 32, 1).
* Camada de Subamostragem 1 (MaxPooling2D): Janela (2x2) para redução dimensional.
* Camada Convolucional 2 (Conv2D): 64 filtros (3x3), Ativação ReLU.
* Camada de Subamostragem 2 (MaxPooling2D): Janela (2x2).
* Achatamento (Flatten): Conversão do mapa de características tridimensional em um vetor linear.
* Camada Densa Oculta (Dense): 64 neurônios com ativação ReLU para combinação de recursos abstratos.
* Camada de Saída (Dense): 3 neurônios com ativação Softmax para gerar a distribuição probabilística das classes.
    - Otimizador: Adam
    - Função de Perda: Sparse Categorical Crossentropy
