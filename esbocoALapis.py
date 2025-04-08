import argparse
import cv2
import os

def aplicar_esboco_lapis(caminho_entrada, caminho_saida):
    try:
        # Processamento da imagem
        imagem = cv2.imread(caminho_entrada)
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}!")
            return False

        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        imagem_desfocada = cv2.GaussianBlur(imagem_cinza, (21, 21), 0)
        
        imagem_cinza_float = imagem_cinza.astype('float32')
        imagem_desfocada_float = imagem_desfocada.astype('float32')
        esboco = cv2.divide(imagem_cinza_float, imagem_desfocada_float + 1e-6, dtype=cv2.CV_32F)
        esboco = (esboco * 255).clip(0, 255).astype('uint8')
        
        # Cria diretório se não existir (com tratamento de erro)
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        
        cv2.imwrite(caminho_saida, esboco)
        return True
        
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transforma imagens em esboços a lápis')
    parser.add_argument('entrada', help='Caminho da imagem de entrada (pasta "Entradas")')
    parser.add_argument('-s', '--saida', help='Nome do arquivo de saída (pasta "Saidas")', default=None)
    
    args = parser.parse_args()
    
    # Configura caminhos
    pasta_entradas = os.path.join(os.path.dirname(__file__), 'Entradas')
    pasta_saidas = os.path.join(os.path.dirname(__file__), 'Saidas')
    
    caminho_entrada = os.path.join(pasta_entradas, args.entrada)
    
    if args.saida:
        caminho_saida = os.path.join(pasta_saidas, args.saida)
    else:
        nome_padrao = f"esboco_{args.entrada}"
        caminho_saida = os.path.join(pasta_saidas, nome_padrao)
    
    # Executa e mostra resultado
    if aplicar_esboco_lapis(caminho_entrada, caminho_saida):
        print(f"✅ Esboço salvo em: {caminho_saida}")
    else:
        print("❌ Falha ao processar a imagem")