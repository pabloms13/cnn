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
