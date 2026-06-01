import numpy as np

# Importações dos módulos anteriores (ajuste os nomes conforme sua estrutura exata)
from network import initialize_parameters
from propagation import forward_pass, compute_loss, backward_pass

def create_mini_batches(X, Y, batch_size):
    """
    Particiona os dados de treinamento em lotes (mini-batches).
    Embaralha os dados sincronicamente para garantir que X e Y permaneçam alinhados.
    
    Argumentos:
    X -- Matriz de dados de entrada de dimensão (784, m)
    Y -- Matriz de rótulos one-hot de dimensão (10, m)
    batch_size -- Inteiro indicando o tamanho de cada lote
    
    Retorna:
    mini_batches -- Lista de tuplas (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    mini_batches = []
    
    # 1. Embaralhamento síncrono das colunas (amostras)
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    # 2. Cálculo do número de lotes completos
    num_complete_minibatches = m // batch_size
    
    # 3. Particionamento dos lotes completos
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * batch_size : (k + 1) * batch_size]
        mini_batch_Y = shuffled_Y[:, k * batch_size : (k + 1) * batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
        
    # 4. Tratamento do último lote (caso m não seja múltiplo do batch_size)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * batch_size : m]
        mini_batches.append((mini_batch_X, mini_batch_Y))
        
    return mini_batches

def update_parameters(params, grads, learning_rate):
    """
    Atualiza os pesos e vieses da rede utilizando o algoritmo de Gradiente Descendente.
    
    Argumentos:
    params -- Dicionário contendo os parâmetros atuais (W1, b1, W2, b2, W3, b3)
    grads -- Dicionário contendo os gradientes calculados no backward_pass (dW1, db1, dW2, db2, dW3, db3)
    learning_rate -- Ponto flutuante determinando o tamanho do passo (estático)
    
    Retorna:
    params -- Dicionário contendo os parâmetros atualizados
    """
    # Como temos 3 camadas (W, b para cada uma), o número de camadas L é len(params) // 2
    L = len(params) // 2
    
    for l in range(1, L + 1):
        # Atualização rígida e constante baseada no gradiente descendente padrão
        params[f'W{l}'] = params[f'W{l}'] - learning_rate * grads[f'dW{l}']
        params[f'b{l}'] = params[f'b{l}'] - learning_rate * grads[f'db{l}']
        
    return params

def calculate_accuracy(predictions_prob, Y_true):
    """
    Função auxiliar para calcular a acurácia.
    
    Argumentos:
    predictions_prob -- Matriz de predições contendo as probabilidades (A3) com formato (10, m)
    Y_true -- Matriz one-hot de rótulos verdadeiros com formato (10, m)
    
    Retorna:
    accuracy -- Ponto flutuante entre 0.0 e 1.0 representando a taxa de acerto
    """
    # Obtém o índice do neurônio com a maior probabilidade/valor (0 a 9)
    predictions = np.argmax(predictions_prob, axis=0)
    labels = np.argmax(Y_true, axis=0)
    
    # Compara as predições com os rótulos reais e calcula a média de acertos
    accuracy = np.mean(predictions == labels)
    return accuracy

def train_network(X_train, Y_train, epochs, learning_rate, batch_size):
    """
    Loop mestre de treinamento da Rede Neural. Orquestra a inicialização, propagação e otimização.
    
    Argumentos:
    X_train -- Matriz de dados de treino transposta (784, m)
    Y_train -- Matriz de rótulos de treino transposta (10, m)
    epochs -- Número exato de iterações sobre todo o conjunto de dados
    learning_rate -- Taxa de aprendizado mantida constante
    batch_size -- Quantidade de amostras processadas por mini-batch
    
    Retorna:
    params -- Parâmetros otimizados após todas as épocas
    """
    # 1. Inicializa parâmetros (Assumindo que a função não precise de argumentos ou utilize tamanhos default. 
    # Se a sua função exigir dimensões como [784, 128, 64, 10], passe-os aqui)
    params = initialize_parameters()
    
    print(f"Iniciando treinamento rígido por {epochs} épocas...")
    print(f"Learning Rate: {learning_rate} | Batch Size: {batch_size}")
    print("-" * 50)
    
    # Loop de épocas (sem interrupções antecipadas)
    for epoch in range(1, epochs + 1):
        
        # Gera os lotes para a época atual (com novo embaralhamento)
        mini_batches = create_mini_batches(X_train, Y_train, batch_size)
        
        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batch
            
            # Etapa A: Propagação do Sinal (Forward Pass)
            A3, cache = forward_pass(mini_batch_X, params)
            
            # Etapa B: Cálculo da Perda (Calculado para obedecer o fluxo solicitado, mas o log fica para o final da época)
            batch_loss = compute_loss(mini_batch_Y, A3)
            
            # Etapa C: Retropropagação do Erro (Backward Pass)
            grads = backward_pass(mini_batch_X, mini_batch_Y, cache, params)
            
            # Etapa D: Otimização (Atualização dos Pesos)
            params = update_parameters(params, grads, learning_rate)
            
        # Avaliação de final de época utilizando todo o conjunto de treino
        A3_train_full, _ = forward_pass(X_train, params)
        epoch_loss = compute_loss(Y_train, A3_train_full)
        epoch_accuracy = calculate_accuracy(A3_train_full, Y_train)
        
        # Feedback no console do Custo (CCE) e da Acurácia de treino
        print(f"Época {epoch:03d}/{epochs} - Custo (CCE): {epoch_loss:.4f} - Acurácia: {epoch_accuracy * 100:.2f}%")
        
    print("-" * 50)
    print("Treinamento concluído.")
    
    return params

if __name__ == "__main__":
    # Exemplo hipotético de como importar seus dados e rodar o script localmente.
    # from data_preprocessing import load_and_preprocess_data
    # X_train, Y_train, X_test, Y_test = load_and_preprocess_data(...)
    
    # parâmetros_otimizados = train_network(
    #    X_train=X_train, 
    #    Y_train=Y_train, 
    #    epochs=50, 
    #    learning_rate=0.01, 
    #    batch_size=64
    # )
    pass