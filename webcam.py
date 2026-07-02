from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo.h5"

# A ordem precisa ser a mesma usada no treino: pastas Imagens/0, Imagens/1, Imagens/2.
NOMES_CLASSES = {
    0: "Chapeu",
    1: "Facao",
    2: "Bota",
}

TAMANHO_IMAGEM = (32, 32)
CONFIANCA_MINIMA = 55.0
ULTIMAS_PREVISOES = 8


def preparar_frame(frame):
    """Aplica o mesmo pre-processamento usado em treinar.py."""
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_equalizado = cv2.equalizeHist(frame_cinza)
    frame_redimensionado = cv2.resize(frame_equalizado, TAMANHO_IMAGEM)

    img = np.array(frame_redimensionado, dtype="float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img


def texto_da_previsao(previsao_media):
    classe_id = int(np.argmax(previsao_media))
    confianca = float(previsao_media[classe_id] * 100)
    nome_classe = NOMES_CLASSES.get(classe_id, f"Classe {classe_id}")

    if confianca < CONFIANCA_MINIMA:
        return "Aproxime um objeto", confianca, (0, 200, 255)

    return f"{nome_classe} ({confianca:.1f}%)", confianca, (0, 255, 0)


def desenhar_resultado(frame, texto, confianca, cor):
    cv2.rectangle(frame, (10, 10), (520, 105), (0, 0, 0), -1)
    cv2.putText(frame, texto, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor, 2)
    cv2.putText(
        frame,
        "Pressione Q para sair",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )

    largura_barra = int(np.clip(confianca, 0, 100) * 3)
    cv2.rectangle(frame, (25, 115), (325, 130), (80, 80, 80), 1)
    cv2.rectangle(frame, (25, 115), (25 + largura_barra, 130), cor, -1)


def main():
    if not MODELO_PATH.exists():
        print(f"Modelo nao encontrado: {MODELO_PATH}")
        print("Execute primeiro: python treinar.py")
        return

    model = load_model(str(MODELO_PATH))
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    historico_previsoes = []
    nomes = ", ".join(NOMES_CLASSES.values())
    print(f"Webcam iniciada. Mostre um destes objetos: {nomes}. Pressione 'q' para sair.")

    while True:
        sucesso, frame = webcam.read()
        if not sucesso:
            print("Nao foi possivel ler a imagem da webcam.")
            break

        img_pronta = preparar_frame(frame)
        previsao = model.predict(img_pronta, verbose=0)[0]

        historico_previsoes.append(previsao)
        if len(historico_previsoes) > ULTIMAS_PREVISOES:
            historico_previsoes.pop(0)

        previsao_media = np.mean(historico_previsoes, axis=0)
        texto, confianca, cor = texto_da_previsao(previsao_media)

        desenhar_resultado(frame, texto, confianca, cor)
        cv2.imshow("Reconhecimento ao vivo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
