from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


# --- 1. CARREGAR E PRE-PROCESSAR OS DADOS ---
BASE_DIR = Path(__file__).resolve().parent
DIRETORIO_BASE = BASE_DIR / "Imagens"
MODELO_PATH = BASE_DIR / "modelo.h5"

# A ordem precisa ser a mesma usada em webcam.py.
CLASSES = ["0", "1", "2"]
TAMANHO_IMAGEM = (32, 32)


def carregar_dados():
    X = []
    y = []

    for classe in CLASSES:
        pasta_classe = DIRETORIO_BASE / classe
        if not pasta_classe.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {pasta_classe}")

        for caminho_imagem in pasta_classe.iterdir():
            if not caminho_imagem.is_file():
                continue

            img = cv2.imread(str(caminho_imagem))
            if img is None:
                continue

            img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_equalizada = cv2.equalizeHist(img_cinza)
            img_redimensionada = cv2.resize(img_equalizada, TAMANHO_IMAGEM)

            X.append(img_redimensionada)
            y.append(int(classe))

    if not X:
        raise RuntimeError("Nenhuma imagem valida foi encontrada em Imagens/0, Imagens/1 e Imagens/2.")

    X = np.array(X, dtype="float32") / 255.0
    X = np.expand_dims(X, -1)
    y = np.array(y)
    return X, y


def criar_modelo():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plotar_historico(historico):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history["accuracy"], label="Treino")
    plt.plot(historico.history["val_accuracy"], label="Validacao")
    plt.title("Acuracia do Modelo")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Validacao")
    plt.title("Perda (Loss) do Modelo")
    plt.legend()

    plt.show()


def main():
    X, y = carregar_dados()
    print(f"Imagens carregadas: {len(X)}")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = criar_modelo()

    print("Treinando o modelo...")
    historico = model.fit(
        X_treino,
        y_treino,
        epochs=15,
        validation_data=(X_teste, y_teste),
        batch_size=8,
    )

    model.save(str(MODELO_PATH))
    print(f"Modelo salvo com sucesso em: {MODELO_PATH}")

    plotar_historico(historico)


if __name__ == "__main__":
    main()
