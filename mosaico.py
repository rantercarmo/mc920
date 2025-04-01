import cv2
import os
import argparse
import numpy as np

def criar_mosaico(caminho_entrada, caminho_saida):
    try:
        # Carrega a imagem em tons de cinza
        imagem = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
            return False

        altura, largura = imagem.shape
        blocos = []

        # Divide a imagem em 16 blocos iguais (4x4)
        for i in range(4):
            for j in range(4):
                y_inicio = i * altura // 4
                y_fim = (i + 1) * altura // 4
                x_inicio = j * largura // 4
                x_fim = (j + 1) * largura // 4
                blocos.append(imagem[y_inicio:y_fim, x_inicio:x_fim])

        # Nova ordem dos blocos conforme a figura (c)
        nova_ordem = [5, 10, 12, 2, 7, 15, 0, 8, 11, 13, 1, 9, 3, 14, 6, 4]
        blocos_reordenados = [blocos[i] for i in nova_ordem]

        # Reconstroi o mosaico
        mosaico = np.vstack([
            np.hstack(blocos_reordenados[0:4]),
            np.hstack(blocos_reordenados[4:8]),
            np.hstack(blocos_reordenados[8:12]),
            np.hstack(blocos_reordenados[12:16])
        ])

        # Garante a pasta de saída existe e salva
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        cv2.imwrite(caminho_saida, mosaico)
        print(f"✅ Mosaico salvo em: {caminho_saida}")
        return True

    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cria mosaico 4x4 com blocos reordenados')
    parser.add_argument('entrada', help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('--saida', '-s', help='Nome personalizado para o arquivo de saída (será salvo na pasta Saídas)', default=None)
    
    args = parser.parse_args()

    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    caminho_entrada = os.path.join(pasta_entradas, args.entrada)

    # Define nome de saída
    if args.saida:
        # Remove extensão se o usuário incluir
        nome_saida = os.path.splitext(args.saida)[0] + '.png'
        caminho_saida = os.path.join(pasta_saidas, nome_saida)
    else:
        # Nome padrão: mosaico_[nome_original].png
        nome_base = os.path.splitext(args.entrada)[0]
        caminho_saida = os.path.join(pasta_saidas, f'mosaico_{nome_base}.png')

    # Executa
    if not criar_mosaico(caminho_entrada, caminho_saida):
        print("Falha ao processar o mosaico")
