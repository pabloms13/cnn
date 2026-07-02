import os
import time
from pathlib import Path

import cv2


# --- CONFIGURACAO ---
# Mude para "0", "1" ou "2" dependendo do objeto que vai gravar.
# A ordem precisa ser a mesma usada em treinar.py e webcam.py.
CLASSE = "0"
TOTAL_FOTOS = 50

BASE_DIR = Path(__file__).resolve().parent
PASTA_DESTINO = BASE_DIR / "Imagens" / CLASSE


def main():
    os.makedirs(PASTA_DESTINO, exist_ok=True)

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    print(f"Prepare o objeto para a classe {CLASSE}...")
    print("A captura comeca em 3 segundos. Mexa o objeto devagar em varios angulos!")
    time.sleep(3)

    contagem = 0
    while contagem < TOTAL_FOTOS:
        sucesso, frame = webcam.read()
        if not sucesso:
            print("Nao foi possivel ler a imagem da webcam.")
            break

        caminho_foto = PASTA_DESTINO / f"foto_{contagem}.jpg"
        cv2.imwrite(str(caminho_foto), frame)

        cv2.putText(
            frame,
            f"Capturando: {contagem + 1}/{TOTAL_FOTOS}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Captura de Dataset", frame)

        contagem += 1
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
    print(f"Pronto! {contagem} fotos salvas em {PASTA_DESTINO}")


if __name__ == "__main__":
    main()
