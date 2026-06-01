import numpy as np

def relu(Z):
    """Função de ativação ReLU."""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Derivada da função ReLU."""
    return (Z > 0).astype(float)

def softmax(Z):
    """
    Função de ativação Softmax.
    Subtrai o valor máximo de Z por coluna para garantir estabilidade numérica
    evitando overflow em exponenciais muito grandes.
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_pass(X, params):
    """
    Realiza a propagação do sinal (Forward Pass) através de 3 camadas.
    
    Argumentos:
    X -- Matriz de dados de entrada de dimensão (784, m)
    params -- Dicionário contendo os pesos e vieses inicializados (W1, b1, W2, b2, W3, b3)
    
    Retorna:
    A3 -- Predições da última camada (Softmax) de dimensão (10, m)
    cache -- Dicionário com os valores de Z e A armazenados para a retropropagação
    """
    # Recuperando parâmetros
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    
    # Camada 1 (Oculta)
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # Camada 2 (Oculta)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    # Camada 3 (Saída)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    
    # Armazenando valores no cache para o backward pass
    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3
    }
    
    return A3, cache

def compute_loss(Y, A3):
    """
    Calcula a função de custo Categorical Cross-Entropy (CCE).
    
    Argumentos:
    Y -- Matriz de rótulos one-hot de dimensão (10, m)
    A3 -- Matriz de predições da rede de dimensão (10, m)
    
    Retorna:
    cost -- Valor do custo médio para o batch
    """
    m = Y.shape[1]
    
    # Adicionando um valor infinitesimal (1e-8) para evitar log(0)
    logprobs = np.log(A3 + 1e-8)
    
    # Cálculo do custo vetorizado
    cost = - (1.0 / m) * np.sum(Y * logprobs)
    
    # Garante que o custo seja retornado como um escalar (ex: 1.5 ao invés de [[1.5]])
    return np.squeeze(cost)

def backward_pass(X, Y, cache, params):
    """
    Realiza a retropropagação do erro (Backward Pass) calculando os gradientes.
    
    Argumentos:
    X -- Matriz de dados de entrada de dimensão (784, m)
    Y -- Matriz de rótulos one-hot de dimensão (10, m)
    cache -- Dicionário contendo os valores de Z e A do forward pass
    params -- Dicionário contendo os pesos atuais da rede
    
    Retorna:
    grads -- Dicionário contendo os gradientes (dW1, db1, dW2, db2, dW3, db3)
    """
    m = X.shape[1]
    
    # Recuperando caches e parâmetros necessários
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    Z1, Z2 = cache['Z1'], cache['Z2']
    W2, W3 = params['W2'], params['W3']
    
    # --- Erro da Camada 3 (Saída) ---
    # Cancelamento algébrico da derivada da CCE com a derivada da Softmax
    dZ3 = A3 - Y
    dW3 = (1.0 / m) * np.dot(dZ3, A2.T)
    db3 = (1.0 / m) * np.sum(dZ3, axis=1, keepdims=True)
    
    # --- Erro da Camada 2 (Oculta) ---
    dZ2 = np.dot(W3.T, dZ3) * relu_derivative(Z2)
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # --- Erro da Camada 1 (Oculta) ---
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    # Empacotando gradientes
    grads = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }
    
    return grads