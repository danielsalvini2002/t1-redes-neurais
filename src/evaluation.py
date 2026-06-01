import numpy as np
# Assumindo que a função forward_pass foi definida no módulo propagation
from propagation import forward_pass 

def predict(X, parameters):
    """
    Executa a inferência cega na rede neural.
    
    Argumentos:
    X -- Matriz de dados de entrada (features, amostras)
    parameters -- Dicionário contendo os pesos e vieses treinados
    
    Retorna:
    predictions -- Vetor contendo a classe predita (0 a 9) para cada amostra
    """
    # Executa o forward_pass. Assumimos que AL é a matriz de saída (Softmax)
    # com dimensões (10, número_de_amostras)
    AL, _ = forward_pass(X, parameters)
    
    # Extrai o índice da classe com maior probabilidade na coluna (axis=0)
    predictions = np.argmax(AL, axis=0)
    
    return predictions

def compute_confusion_matrix(Y_true, Y_pred, num_classes=10):
    """
    Constrói a matriz de confusão multiclasse a partir do zero.
    """
    # Reverte o Y_true caso ele esteja em formato One-Hot Encoding
    if Y_true.ndim > 1 and Y_true.shape[0] == num_classes:
        y_true_indices = np.argmax(Y_true, axis=0)
    elif Y_true.ndim > 1 and Y_true.shape[1] == num_classes:
        y_true_indices = np.argmax(Y_true, axis=1)
    else:
        y_true_indices = Y_true.flatten()
        
    y_pred_indices = Y_pred.flatten()
    
    # Inicializa a matriz 10x10 com zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Preenche a matriz iterando sobre as amostras
    # i (linha) = classe verdadeira, j (coluna) = classe predita
    for true_idx, pred_idx in zip(y_true_indices, y_pred_indices):
        conf_matrix[true_idx, pred_idx] += 1
        
    return conf_matrix

def calculate_metrics(conf_matrix):
    """
    Extrai as métricas de Acurácia Global, Precisão, Recall e F1-Score.
    Implementa estabilidade numérica no denominador (epsilon).
    """
    epsilon = 1e-8
    
    # Verdadeiros Positivos (TP) são os elementos da diagonal principal
    TP = np.diag(conf_matrix)
    
    # Acurácia Global: Soma da diagonal dividida pelo total absoluto
    total_samples = np.sum(conf_matrix)
    accuracy = np.sum(TP) / (total_samples + epsilon)
    
    # Precisão por classe (TP / TP + FP) -> Soma ao longo das colunas (axis=0)
    col_sums = np.sum(conf_matrix, axis=0)
    precision = TP / (col_sums + epsilon)
    
    # Recall por classe (TP / TP + FN) -> Soma ao longo das linhas (axis=1)
    row_sums = np.sum(conf_matrix, axis=1)
    recall = TP / (row_sums + epsilon)
    
    # F1-Score por classe: 2 * (P * R) / (P + R)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return accuracy, precision, recall, f1_score

def evaluate_model(X_test, Y_test, parameters):
    """
    Fluxo consolidador: Avalia o modelo e imprime o relatório analítico tabulado.
    """
    # 1. Inferência
    predictions = predict(X_test, parameters)
    
    # 2. Construção da Matriz
    conf_matrix = compute_confusion_matrix(Y_test, predictions, num_classes=10)
    
    # 3. Cálculo das Métricas
    accuracy, precision, recall, f1_score = calculate_metrics(conf_matrix)
    
    # 4. Cálculo das Médias Macro (Macro Average)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)
    
    # ==========================================
    # Exibição do Relatório
    # ==========================================
    print("="*60)
    print(" MATRIZ DE CONFUSÃO (Linha: Target | Coluna: Predição)")
    print("="*60)
    
    # Cabeçalho numérico da matriz
    header = "    " + "".join([f"{i:5d}" for i in range(10)])
    print(header)
    print("-" * len(header))
    
    # Linhas da matriz com tabulação
    for i in range(10):
        row_str = f"{i} | " + "".join([f"{conf_matrix[i, j]:5d}" for j in range(10)])
        print(row_str)
        
    print("\n" + "="*60)
    print(" RELATÓRIO DE MÉTRICAS POR CLASSE (0-9)")
    print("="*60)
    print(f"Classe | Precision |  Recall  | F1-Score")
    print("-" * 45)
    for i in range(10):
        print(f"  {i:2d}   |  {precision[i]:.4f}   |  {recall[i]:.4f}  |  {f1_score[i]:.4f}")
        
    print("\n" + "="*60)
    print(" MÉTRICAS GLOBAIS / MACRO AVERAGE")
    print("="*60)
    print(f"Acurácia Global  : {accuracy:.4f}")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print("="*60)