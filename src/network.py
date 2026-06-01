import numpy as np

# =====================================================================
# INICIALIZAÇÃO DE PARÂMETROS
# =====================================================================

def initialize_parameters():
    """
    Inicializa os pesos e vieses para uma rede neural feedforward de 3 camadas
    (2 ocultas e 1 de saída) com a arquitetura [784 -> 10 -> 10 -> 10].
    
    Aplica a Inicialização de He para os pesos (W) e inicializa vieses (b) com zero.
    
    Retorna:
    parameters -- dicionário python contendo:
                  W1: matriz de pesos da camada 1, shape (10, 784)
                  b1: vetor de viés da camada 1, shape (10, 1)
                  W2: matriz de pesos da camada 2, shape (10, 10)
                  b2: vetor de viés da camada 2, shape (10, 1)
                  W3: matriz de pesos da camada de saída, shape (10, 10)
                  b3: vetor de viés da camada de saída, shape (10, 1)
    """
    # Definição das dimensões da arquitetura
    n_x = 784  # Tamanho da camada de entrada (pixels achatados)
    n_h1 = 10  # Tamanho da primeira camada oculta
    n_h2 = 10  # Tamanho da segunda camada oculta
    n_y = 10   # Tamanho da camada de saída (10 classes do MNIST)
    
    # Camada 1
    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h1, 1))
    
    # Camada 2
    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2.0 / n_h1)
    b2 = np.zeros((n_h2, 1))
    
    # Camada 3 (Saída)
    W3 = np.random.randn(n_y, n_h2) * np.sqrt(2.0 / n_h2)
    b3 = np.zeros((n_y, 1))
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    
    return parameters

# =====================================================================
# FUNÇÕES DE ATIVAÇÃO E DERIVADAS
# =====================================================================

def relu(Z):
    """
    Implementa a função de ativação Rectified Linear Unit (ReLU).
    Retorna o valor máximo entre 0 e Z de forma element-wise.
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Calcula a derivada da função ReLU.
    Retorna 1 para os elementos onde Z > 0, e 0 caso contrário.
    """
    # Z > 0 cria uma matriz booleana. astype(float) converte True para 1.0 e False para 0.0
    return (Z > 0).astype(float)

def softmax(Z):
    """
    Implementa a função Softmax com estabilidade numérica.
    Calcula as probabilidades das classes ao longo das colunas (amostras).
    """
    # Subtrai o valor máximo de cada coluna para estabilidade numérica (evita overflow de np.exp)
    # keepdims=True garante que o vetor resultante continue com a segunda dimensão, permitindo o broadcasting correto
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    
    # Exponencial de Z estabilizado
    exp_Z = np.exp(Z_shifted)
    
    # Divide pela soma das exponenciais de cada coluna (cada amostra)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    return A

# =====================================================================
# TESTES BÁSICOS (Executado apenas se o script for rodado diretamente)
# =====================================================================
if __name__ == "__main__":
    params = initialize_parameters()
    print("=== TESTE DE INICIALIZAÇÃO ===")
    for key, value in params.items():
        print(f"{key} shape: {value.shape}")
        
    print("\n=== TESTE SOFTMAX (Estabilidade Numérica) ===")
    # Simulando um Z de saída com valores excessivamente altos para 3 exemplos (colunas)
    Z_test = np.array([[1000, 2000, 3000], 
                       [1001, 2001, 3001], 
                       [999,  1999, 2999]])
    
    A_test = softmax(Z_test)
    print("Probabilidades computadas sem overflow:\n", A_test)
    print("Soma das probabilidades por coluna (deve ser 1):", np.sum(A_test, axis=0))