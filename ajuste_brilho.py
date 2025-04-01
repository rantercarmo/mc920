import cv2
import os
import argparse
import numpy as np

def ajuste_gamma(imagem, gamma):
    """Aplica correção gamma na imagem"""
    # Normaliza para [0, 1]
    imagem_norm = imagem.astype('float32') / 255.0
    
    # Aplica correção gamma
    imagem_corrigida = np.power(imagem_norm, 1.0/gamma)
    
    # Retorna para [0, 255] e converte para uint8
    return (imagem_corrigida * 255).clip(0, 255).astype('uint8')

def processar_imagem(caminho_entrada, caminho_saida_base, gammas):
    try:
        # Carrega a imagem (converte para tons de cinza se for colorida)
        imagem = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
            return False

        # Processa cada valor de gamma
        for gamma in gammas:
            # Aplica correção
            imagem_corrigida = ajuste_gamma(imagem, gamma)
            
            # Salva o resultado
            nome_saida = f"{os.path.splitext(os.path.basename(caminho_saida_base))[0]}_gamma{gamma}.png"
            caminho_saida = os.path.join(os.path.dirname(caminho_saida_base), nome_saida)
            
            os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
            cv2.imwrite(caminho_saida, imagem_corrigida)
            print(f"✅ Imagem com γ={gamma} salva em: {caminho_saida}")
        
        return True
        
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aplica correção gamma para ajuste de brilho')
    parser.add_argument('entrada', help='Nome do arquivo na pasta Entradas (ex: imagem.png)')
    parser.add_argument('-s', '--saida', help='Nome base do arquivo de saída (sem extensão)', default='brilho')
    
    args = parser.parse_args()
    
    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    
    caminho_entrada = os.path.join(pasta_entradas, args.entrada)
    caminho_saida_base = os.path.join(pasta_saidas, args.saida)
    
    # Valores de gamma a serem testados (conforme enunciado)
    gammas = [1.5, 2.5, 3.5]
    
    # Executa o processamento
    if not processar_imagem(caminho_entrada, caminho_saida_base, gammas):
        print("❌ Falha ao processar a imagem")