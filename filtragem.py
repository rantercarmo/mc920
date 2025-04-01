import cv2
import os
import argparse
import numpy as np
from math import sqrt

def carregar_imagem(caminho_entrada):
    """Carrega a imagem em tons de cinza e normaliza para float32 no intervalo [0,1]"""
    imagem = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
        return None
    return imagem.astype(np.float32) / 255.0

def criar_filtro(filtro_id):
    """Retorna o kernel do filtro especificado"""
    filtros = {
        'h1': np.array([  # Laplaciano modificado
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ]),
        'h2': (1/256) * np.array([  # Gaussiano 5x5
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ]),
        'h3': np.array([  # Sobel horizontal
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        'h4': np.array([  # Sobel vertical
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        'h5': np.array([  # Laplaciano
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]),
        'h6': (1/9) * np.ones((3,3)),  # Média 3x3
        'h7': np.array([  # Filtro de Prewitt horizontal
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]),
        'h8': np.array([  # Filtro de Prewitt vertical
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]),
        'h9': (1/9) * np.ones((3,3)),  # Média 3x3 (igual ao h6)
        'h10': (1/8) * np.array([  # Aguçamento modificado
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1]
        ]),
        'h11': np.array([  # Embossing
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]
        ])
    }
    return filtros.get(filtro_id)

def aplicar_filtro(imagem, filtro_id):
    """
    Aplica o filtro especificado na imagem
    Args:
        imagem: imagem normalizada [0,1]
        filtro_id: identificador do filtro (h1 a h11)
    Returns:
        Imagem filtrada em formato uint8 [0,255] e explicação do filtro
    """
    img_8bit = (imagem * 255).astype(np.uint8)
    
    if filtro_id == 'sobel_combined':
        sobel_x = cv2.filter2D(img_8bit, -1, criar_filtro('h3'))
        sobel_y = cv2.filter2D(img_8bit, -1, criar_filtro('h4'))
        combined = np.sqrt(np.square(sobel_x.astype(np.float32)) + 
                         np.square(sobel_y.astype(np.float32)))
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
        return combined.astype(np.uint8), "Magnitude do gradiente (Sobel combinado): detecção de bordas em todas as direções"
    
    kernel = criar_filtro(filtro_id)
    if kernel is not None:
        filtrada = cv2.filter2D(img_8bit, -1, kernel)
        
        if filtro_id in ['h1', 'h3', 'h4', 'h5', 'h7', 'h8', 'h11']:
            filtrada = cv2.normalize(filtrada, None, 0, 255, cv2.NORM_MINMAX)
        
        explicacao = {
            'h1': "Laplaciano modificado: realce de bordas com maior sensibilidade",
            'h2': "Gaussiano 5x5: suavização forte com redução de ruído",
            'h3': "Sobel horizontal: destaca bordas verticais",
            'h4': "Sobel vertical: destaca bordas horizontais",
            'h5': "Laplaciano: detecção de bordas em todas as direções",
            'h6': "Média 3x3: suavização simples para redução de ruído",
            'h7': "Prewitt horizontal: detecção de bordas verticais (similar a Sobel)",
            'h8': "Prewitt vertical: detecção de bordas horizontais (similar a Sobel)",
            'h9': "Média 3x3: igual ao h6, suavização simples",
            'h10': "Aguçamento modificado: realce de detalhes finos e bordas",
            'h11': "Embossing: efeito de relevo com iluminação a 45 graus"
        }.get(filtro_id, "Filtro desconhecido")
        
        return filtrada.astype(np.uint8), explicacao
    
    return None, ""

def aplicar_todos_filtros(imagem, pasta_saida, nome_base):
    """Aplica todos os filtros e salva os resultados"""
    filtros = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'sobel_combined']
    resultados = []
    
    for filtro in filtros:
        imagem_filtrada, explicacao = aplicar_filtro(imagem, filtro)
        if imagem_filtrada is not None:
            caminho_saida = os.path.join(pasta_saida, f'filtrada_{filtro}_{nome_base}.png')
            salvar_imagem(caminho_saida, imagem_filtrada)
            resultados.append((filtro, explicacao))
    
    return resultados

def salvar_imagem(caminho_saida, imagem):
    """Salva a imagem no caminho especificado"""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aplica filtros em imagens monocromáticas')
    parser.add_argument('entrada', help='Nome da imagem na pasta Entradas (ex: foto.jpg)')
    parser.add_argument('--filtro', '-f', 
                       choices=['all', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 
                               'h9', 'h10', 'h11', 'sobel_combined'],
                       default='all', help='Filtro a ser aplicado (ou "all" para todos)')
    parser.add_argument('--saida', '-s', help='Nome personalizado para o arquivo de saída', default=None)
    
    args = parser.parse_args()
    
    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    caminho_entrada = os.path.join(pasta_entradas, args.entrada)
    
    # Processa a imagem
    imagem = carregar_imagem(caminho_entrada)
    if imagem is not None:
        nome_base = os.path.splitext(args.entrada)[0]
        
        if args.filtro == 'all':
            print("Aplicando todos os filtros...")
            resultados = aplicar_todos_filtros(imagem, pasta_saidas, nome_base)
            
            print("\nResumo dos filtros aplicados:")
            for filtro, explicacao in resultados:
                print(f"- {filtro}: {explicacao}")
            print(f"\n✅ Todas as imagens filtradas foram salvas na pasta 'Saidas'")
        else:
            imagem_filtrada, explicacao = aplicar_filtro(imagem, args.filtro)
            if imagem_filtrada is not None:
                print(f"Efeito do filtro {args.filtro}: {explicacao}")
                
                if args.saida:
                    nome_saida = os.path.splitext(args.saida)[0] + '.png'
                else:
                    nome_saida = f'filtrada_{args.filtro}_{nome_base}.png'
                
                caminho_saida = os.path.join(pasta_saidas, nome_saida)
                salvar_imagem(caminho_saida, imagem_filtrada)
                print(f"✅ Imagem filtrada salva em: {caminho_saida}")