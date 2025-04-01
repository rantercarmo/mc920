import cv2
import numpy as np

def aplicar_esboco_lapis(caminho_entrada, caminho_saida):
    # 1. Ler a imagem (PNG)
    imagem = cv2.imread(caminho_entrada)
    
    if imagem is None:
        print("Erro: Não foi possível carregar a imagem. Verifique o caminho.")
        return
    
    # 2. Converter para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # 3. Aplicar filtro gaussiano (21x21)
    imagem_desfocada = cv2.GaussianBlur(imagem_cinza, (21, 21), 0)
    
    # 4. Dividir a imagem original pela versão desfocada para realçar contornos
    # Adicionamos um pequeno valor (1e-6) para evitar divisão por zero
    # Converta ambas as imagens para float32 e especifique o tipo de saída
    imagem_cinza_float = imagem_cinza.astype('float32')
    imagem_desfocada_float = imagem_desfocada.astype('float32')
    esboco = cv2.divide(imagem_cinza_float, imagem_desfocada_float + 1e-6, dtype=cv2.CV_32F)
    esboco = (esboco * 255).clip(0, 255).astype('uint8')  # Converta de volta para uint8
    
    # Inverter a imagem para obter o efeito de esboço a lápis
   # esboco = 255 - esboco
    
    # 5. Salvar o resultado (PNG)
    cv2.imwrite(caminho_saida, esboco)
    print(f"Esboço a lápis salvo com sucesso em: {caminho_saida}")

# Exemplo de uso:
# Substitua 'entrada.png' pelo caminho da sua imagem de entrada
# e 'saida.png' pelo caminho onde deseja salvar o resultado
aplicar_esboco_lapis('/Users/huntersoaresdocarmo/Desktop/helio/Entradas/watch.png', '/Users/huntersoaresdocarmo/Desktop/helio/Saidas/saidatask1.png')