import cv2
import os
import argparse
import numpy as np

def carregar_imagem(caminho_entrada):
    imagem = cv2.imread(caminho_entrada)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
        return None
    return imagem.astype(np.float32) / 255.0

def aplicar_transformacao_cor(imagem):
    # Converte de BGR para RGB
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Matriz de transformação (efeito sépia)
    matriz_transformacao = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Aplica a transformação
    transformada = np.dot(imagem_rgb, matriz_transformacao.T)
    
    # Clipa os valores para o intervalo [0,1] e converte para 8 bits
    transformada = np.clip(transformada, 0, 1)
    transformada = (transformada * 255).astype(np.uint8)

    # Converte de volta para BGR antes de salvar
    return cv2.cvtColor(transformada, cv2.COLOR_RGB2BGR)

def salvar_imagem(caminho_saida, imagem):
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem)
    print(f"✅ Imagem transformada salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aplica transformação de cores usando matriz de conversão')
    parser.add_argument('entrada', help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('--saida', '-s', help='Nome personalizado para o arquivo de saída (será salvo na pasta Saídas)', default=None)
    
    args = parser.parse_args()
    
    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    caminho_entrada = os.path.join(pasta_entradas, args.entrada)
    
    # Define nome de saída
    if args.saida:
        nome_saida = os.path.splitext(args.saida)[0] + '.png'
    else:
        nome_base = os.path.splitext(args.entrada)[0]
        nome_saida = f'transformada_{nome_base}.png'
    caminho_saida = os.path.join(pasta_saidas, nome_saida)
    
    # Processa a imagem
    imagem = carregar_imagem(caminho_entrada)
    if imagem is not None:
        imagem_transformada = aplicar_transformacao_cor(imagem)
        salvar_imagem(caminho_saida, imagem_transformada)