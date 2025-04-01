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

def combinar_imagens(imagem_a, imagem_b, peso_a):
    """
    Combina duas imagens monocromáticas usando média ponderada
    Args:
        imagem_a: primeira imagem normalizada [0,1]
        imagem_b: segunda imagem normalizada [0,1]
        peso_a: peso da imagem A (0 a 1)
    Returns:
        Imagem combinada em formato uint8 [0,255]
    """
    # Verifica se as imagens têm o mesmo tamanho
    if imagem_a.shape != imagem_b.shape:
        print("Erro: As imagens devem ter o mesmo tamanho!")
        return None
    
    # Calcula o peso da imagem B
    peso_b = 1.0 - peso_a
    
    # Combinação ponderada
    combinada = peso_a * imagem_a + peso_b * imagem_b
    
    # Clipa e converte para 8 bits
    combinada = np.clip(combinada, 0, 1)
    return (combinada * 255).astype(np.uint8)

def salvar_imagem(caminho_saida, imagem):
    """Salva a imagem no caminho especificado"""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem)
    print(f"✅ Imagem combinada salva em: {caminho_saida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combina duas imagens monocromáticas usando média ponderada')
    parser.add_argument('entrada_a', help='Nome da primeira imagem na pasta Entradas (ex: foto1.jpg)')
    parser.add_argument('entrada_b', help='Nome da segunda imagem na pasta Entradas (ex: foto2.jpg)')
    parser.add_argument('--peso_a', '-p', type=float, default=0.5,
                        help='Peso da primeira imagem (0 a 1). Ex: 0.2 para 20%% da imagem A', metavar='PESO')
    parser.add_argument('--saida', '-s', help='Nome personalizado para o arquivo de saída (será salvo na pasta Saídas)', default=None)
    
    args = parser.parse_args()
    
    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    caminho_entrada_a = os.path.join(pasta_entradas, args.entrada_a)
    caminho_entrada_b = os.path.join(pasta_entradas, args.entrada_b)
    
    # Define nome de saída
    if args.saida:
        nome_saida = os.path.splitext(args.saida)[0] + '.png'
    else:
        nome_base_a = os.path.splitext(args.entrada_a)[0]
        nome_base_b = os.path.splitext(args.entrada_b)[0]
        peso_str = str(args.peso_a).replace('.', '_')  # Substitui ponto por underscore
        nome_saida = f'combinada_{peso_str}A_{nome_base_a}_{nome_base_b}.png'
    caminho_saida = os.path.join(pasta_saidas, nome_saida)
    
    # Processa as imagens
    imagem_a = carregar_imagem(caminho_entrada_a)
    imagem_b = carregar_imagem(caminho_entrada_b)
    
    if imagem_a is not None and imagem_b is not None:
        imagem_combinada = combinar_imagens(imagem_a, imagem_b, args.peso_a)
        if imagem_combinada is not None:
            salvar_imagem(caminho_saida, imagem_combinada)