import cv2
import os
import argparse
import numpy as np

def carregar_imagem(caminho_entrada):
    """Carrega a imagem em tons de cinza e normaliza para float32 no intervalo [0,1]"""
    imagem = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
        return None
    return imagem.astype(np.float32) / 255.0

def aplicar_transformacoes(imagem, transformacao):
    """
    Aplica diferentes transformações de intensidade na imagem
    Args:
        imagem: imagem normalizada [0,1]
        transformacao: tipo de transformação a aplicar ('negativo', 'intervalo', 'inverter_pares', 
                      'reflexao_linhas', 'espelhamento_vertical')
    Returns:
        Imagem transformada em formato uint8 [0,255]
    """
    # Converte para valores de 8 bits (0-255)
    img_8bit = (imagem * 255).astype(np.uint8)
    
    if transformacao == 'negativo':
        # (b) Negativo da imagem
        transformada = 255 - img_8bit
    
    elif transformacao == 'intervalo':
        # (c) Converter para intervalo [100, 200]
        min_original = np.min(img_8bit)
        max_original = np.max(img_8bit)
        transformada = ((img_8bit - min_original) / (max_original - min_original)) * 100 + 100
        transformada = transformada.astype(np.uint8)
    
    elif transformacao == 'inverter_pares':
        # (d) Inverter linhas pares
        transformada = img_8bit.copy()
        transformada[::2, :] = transformada[::2, ::-1]  # Inverte linhas pares
    
    elif transformacao == 'reflexao_linhas':
        # (e) Espelhar metade superior na inferior
        transformada = img_8bit.copy()
        h = transformada.shape[0]
        metade = h // 2
        transformada[metade:, :] = transformada[metade-1::-1, :]  # Espelha a metade superior
    
    elif transformacao == 'espelhamento_vertical':
        # (f) Espelhamento vertical completo
        transformada = cv2.flip(img_8bit, 0)
    
    else:
        print("Transformação desconhecida!")
        return None
    
    return transformada

def salvar_imagem(caminho_saida, imagem):
    """Salva a imagem no caminho especificado"""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem)
    print(f"✅ Imagem transformada salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aplica transformações de intensidade em imagens monocromáticas')
    parser.add_argument('entrada', help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('--transformacao', '-t', 
                        choices=['negativo', 'intervalo', 'inverter_pares', 'reflexao_linhas', 'espelhamento_vertical'],
                        required=True, help='Tipo de transformação a aplicar')
    parser.add_argument('--saida', '-s', help='Nome personalizado para o arquivo de saída', default=None)
    
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
        nome_saida = f'transformada_{args.transformacao}_{nome_base}.png'
    caminho_saida = os.path.join(pasta_saidas, nome_saida)
    
    # Processa a imagem
    imagem = carregar_imagem(caminho_entrada)
    if imagem is not None:
        imagem_transformada = aplicar_transformacoes(imagem, args.transformacao)
        if imagem_transformada is not None:
            salvar_imagem(caminho_saida, imagem_transformada)