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
        transformacao: tipo de transformação a aplicar
    Returns:
        Imagem transformada em formato uint8 [0,255] ou None em caso de erro
    """
    try:
        # Converte para valores de 8 bits (0-255)
        img_8bit = (imagem * 255).astype(np.uint8)
        transformada = img_8bit.copy()

        if transformacao == 'negativo':
            transformada = 255 - img_8bit
        
        elif transformacao == 'intervalo':
            min_val, max_val = np.min(img_8bit), np.max(img_8bit)
            if max_val == min_val:  # Evita divisão por zero
                transformada = np.full_like(img_8bit, 150)  # Valor médio se todos pixels forem iguais
            else:
                transformada = ((img_8bit - min_val) / (max_val - min_val)) * 100 + 100
                transformada = transformada.astype(np.uint8)
        
        elif transformacao == 'inverter_pares':
            transformada[::2, :] = transformada[::2, ::-1]  # Inverte linhas pares
        
        elif transformacao == 'reflexao_linhas':
            h, w = transformada.shape
            metade = (h + 1) // 2  # Arredonda para cima para pegar a linha do meio na parte superior
            parte_superior = transformada[:metade]
            # Ajusta o tamanho da parte inferior para combinar com o espelhamento
            transformada[-metade:] = parte_superior[::-1][-metade:]  # Espelha a metade superior
        
        elif transformacao == 'espelhamento_vertical':
            transformada = cv2.flip(img_8bit, 0)
        
        else:
            raise ValueError(f"Transformação desconhecida: {transformacao}")
        
        return transformada

    except Exception as e:
        print(f"Erro durante a transformação {transformacao}: {str(e)}")
        return None

def salvar_imagem(caminho_saida, imagem):
    """Salva a imagem no caminho especificado"""
    try:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        if not cv2.imwrite(caminho_saida, imagem):
            raise IOError(f"Falha ao salvar imagem em {caminho_saida}")
        print(f"✅ Imagem transformada salva em: {caminho_saida}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar imagem: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aplica transformações de intensidade em imagens monocromáticas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('entrada', 
                       help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('-t', '--transformacao', 
                       choices=['negativo', 'intervalo', 'inverter_pares', 
                               'reflexao_linhas', 'espelhamento_vertical'],
                       required=True,
                       help='Tipo de transformação a aplicar')
    parser.add_argument('-s', '--saida', 
                       help='Nome personalizado para o arquivo de saída (sem extensão)',
                       default=None)
    
    args = parser.parse_args()

    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    caminho_entrada = os.path.join(pasta_entradas, args.entrada)
    
    # Processa a imagem
    imagem = carregar_imagem(caminho_entrada)
    if imagem is None:
        exit(1)

    imagem_transformada = aplicar_transformacoes(imagem, args.transformacao)
    if imagem_transformada is None:
        exit(1)

    # Define nome de saída
    nome_base = os.path.splitext(args.entrada)[0]
    nome_saida = f"{args.saida or f'transformada_{args.transformacao}_{nome_base}'}.png"
    caminho_saida = os.path.join(pasta_saidas, nome_saida)

    if not salvar_imagem(caminho_saida, imagem_transformada):
        exit(1)