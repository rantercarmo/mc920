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

def quantizar_imagem(imagem, niveis):
    """
    Quantiza a imagem para um número específico de níveis de cinza
    Args:
        imagem: imagem normalizada [0,1]
        niveis: número de níveis de quantização (2, 4, 8, 16, 32, 64, 256)
    Returns:
        Imagem quantizada em formato uint8 [0,255]
    """
    # Converte para valores de 8 bits (0-255)
    img_8bit = (imagem * 255).astype(np.uint8)
    
    # Calcula o tamanho do intervalo de quantização
    if niveis == 256:
        return img_8bit  # Sem quantização
    
    if niveis < 2 or niveis > 256 or not (niveis & (niveis - 1) == 0):
        print("Erro: O número de níveis deve ser potência de 2 entre 2 e 256!")
        return None
    
    # Calcula o fator de quantização
    fator = 256 / niveis
    
    # Quantiza a imagem
    quantizada = (img_8bit / fator).astype(np.uint8) * fator
    
    # Para níveis extremos (2 níveis), ajustamos para preto e branco puro
    if niveis == 2:
        quantizada = np.where(quantizada > 127, 255, 0)
    
    return quantizada.astype(np.uint8)

def salvar_imagem(caminho_saida, imagem):
    """Salva a imagem no caminho especificado"""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem)
    print(f"✅ Imagem quantizada salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantiza uma imagem monocromática em diferentes níveis de cinza')
    parser.add_argument('entrada', help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('--niveis', '-n', type=int, 
                        choices=[2, 4, 8, 16, 32, 64, 256],
                        required=True, help='Número de níveis de quantização (2, 4, 8, 16, 32, 64, 256)')
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
        nome_saida = f'quantizada_{args.niveis}niveis_{nome_base}.png'
    caminho_saida = os.path.join(pasta_saidas, nome_saida)
    
    # Processa a imagem
    imagem = carregar_imagem(caminho_entrada)
    if imagem is not None:
        imagem_quantizada = quantizar_imagem(imagem, args.niveis)
        if imagem_quantizada is not None:
            salvar_imagem(caminho_saida, imagem_quantizada)