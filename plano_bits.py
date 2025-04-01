import cv2
import os
import argparse
import numpy as np

def carregar_imagem(caminho_entrada):
    """Carrega a imagem e normaliza para float32 no intervalo [0,1]"""
    imagem = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)  # Carrega diretamente em tons de cinza
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
        return None
    return imagem.astype(np.float32) / 255.0

def extrair_planos_bits(imagem, plano):
    """
    Extrai um plano de bits específico da imagem monocromática
    Args:
        imagem: imagem em tons de cinza normalizada [0,1]
        plano: número do plano de bit a extrair (0 a 7)
    Returns:
        Imagem binária contendo apenas o plano de bit solicitado
    """
    # Converte para valores de 8 bits (0-255)
    imagem_8bit = (imagem * 255).astype(np.uint8)
    
    # Cria máscara para o plano de bit solicitado
    mascara = 1 << plano
    
    # Aplica a máscara e normaliza para 0 ou 1
    plano_bit = (imagem_8bit & mascara) >> plano
    
    # Converte para 0 ou 255 para melhor visualização
    plano_bit = plano_bit * 255
    
    return plano_bit.astype(np.uint8)

def salvar_imagem(caminho_saida, imagem):
    """Salva a imagem no caminho especificado"""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem)
    print(f"✅ Imagem transformada salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrai planos de bits de uma imagem monocromática')
    parser.add_argument('entrada', help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('--plano', '-p', type=int, choices=range(0, 8), 
                        help='Plano de bit a extrair (0 a 7)', required=True)
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
        nome_saida = f'plano_bit_{args.plano}_{nome_base}.png'
    caminho_saida = os.path.join(pasta_saidas, nome_saida)
    
    # Processa a imagem
    imagem = carregar_imagem(caminho_entrada)
    if imagem is not None:
        plano_bit = extrair_planos_bits(imagem, args.plano)
        salvar_imagem(caminho_saida, plano_bit)