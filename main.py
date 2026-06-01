import os
import sys

# Adiciona a pasta 'src' ao caminho do sistema para permitir as importações dos módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import get_preprocessed_data
from train import train_network
from evaluation import evaluate_model

def main():
    print("="*60)
    print(" INÍCIO DO PIPELINE DA REDE NEURAL - MNIST")
    print("="*60)

    # 1. Definição dos caminhos dos ficheiros do dataset MNIST
    # Assumimos que os ficheiros descompactados estão numa pasta 'data' na raiz do projeto
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    
    train_img_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_lbl_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_img_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_lbl_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    # Verificação rápida se os ficheiros de dados existem
    if not os.path.exists(train_img_path):
        print(f"[ERRO] O ficheiro de dados não foi encontrado em: {train_img_path}")
        print("Por favor, certifique-se de que os ficheiros originais do MNIST estão na pasta 'data/'.")
        return

    # 2. Configuração dos Hiperparâmetros
    # (Pode ajustar estes valores consoante a necessidade)
    EPOCHS = 30
    LEARNING_RATE = 0.1
    BATCH_SIZE = 128

    # 3. Pré-processamento dos Dados
    print("\n[1/3] A carregar e pré-processar os dados do MNIST...")
    X_train, Y_train, X_test, Y_test = get_preprocessed_data(
        train_img_path=train_img_path,
        train_lbl_path=train_lbl_path,
        test_img_path=test_img_path,
        test_lbl_path=test_lbl_path
    )
    print(f"Dados de treino carregados: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Dados de teste carregados: X={X_test.shape}, Y={Y_test.shape}")

    # 4. Treino da Rede Neural
    print("\n[2/3] A iniciar o processo de treino do modelo...")
    parametros_otimizados = train_network(
        X_train=X_train, 
        Y_train=Y_train, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE, 
        batch_size=BATCH_SIZE
    )

    # 5. Avaliação do Modelo
    print("\n[3/3] A avaliar o modelo com o conjunto de teste...")
    evaluate_model(X_test, Y_test, parametros_otimizados)
    
    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()