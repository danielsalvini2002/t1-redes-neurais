import numpy as np # linear algebra
import struct
from array import array
from os.path import join

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
 
    def read_images_labels(self, images_filepath, labels_filepath):  
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())  
 
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())  
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img  
 
        return images, labels
 
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

# =====================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO (NUMPY)
# =====================================================================

def flatten_and_normalize(images_list):
    """
    Converte a lista de imagens, achata de (m, 28, 28) para (m, 784),
    normaliza para [0, 1] e transpõe para (784, m).
    """
    # Converter para array do NumPy (m, 28, 28)
    m = len(images_list)
    images_array = np.array(images_list)
    
    # Achatar para (m, 784)
    images_flattened = images_array.reshape(m, -1)
    
    # Normalizar dividindo por 255.0
    images_normalized = images_flattened / 255.0
    
    # Transpor para que cada coluna seja um exemplo: shape final (784, m)
    return images_normalized.T

def one_hot_encode(labels_list, num_classes=10):
    """
    Converte os rótulos de inteiros para matriz one-hot e transpõe para (10, m).
    """
    labels_array = np.array(labels_list)
    m = labels_array.shape[0]
    
    # Matriz esparsa de zeros (m, 10)
    one_hot = np.zeros((m, num_classes))
    
    # Preenche com 1 na posição da classe correspondente
    one_hot[np.arange(m), labels_array] = 1
    
    # Transpor para shape final (10, m)
    return one_hot.T

def get_preprocessed_data(train_img_path, train_lbl_path, test_img_path, test_lbl_path):
    """
    Centraliza o fluxo de leitura e pré-processamento, 
    retornando as 4 matrizes prontas para a rede neural.
    """
    # 1. Instanciar o leitor
    dataloader = MnistDataloader(train_img_path, train_lbl_path, test_img_path, test_lbl_path)
    
    # 2. Ler dados brutos
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = dataloader.load_data()
    
    # 3. Processar imagens (Achatamento + Normalização + Transposição)
    X_train = flatten_and_normalize(x_train_raw)
    X_test  = flatten_and_normalize(x_test_raw)
    
    # 4. Processar rótulos (One-Hot Encoding + Transposição)
    Y_train = one_hot_encode(y_train_raw)
    Y_test  = one_hot_encode(y_test_raw)
    
    return X_train, Y_train, X_test, Y_test

# Exemplo de utilização (Testando a saída):
if __name__ == "__main__":
    # Ajuste os caminhos relativos de acordo com os arquivos na sua pasta 'data/'
    X_train, Y_train, X_test, Y_test = get_preprocessed_data(
        train_img_path='../data/train-images-idx3-ubyte',
        train_lbl_path='../data/train-labels-idx1-ubyte',
        test_img_path='../data/t10k-images-idx3-ubyte',
        test_lbl_path='../data/t10k-labels-idx1-ubyte'
    )
    
    print("=== DADOS PRÉ-PROCESSADOS COM SUCESSO ===")
    print(f"Shape X_train: {X_train.shape} -> (784 features, 60000 amostras)")
    print(f"Shape Y_train: {Y_train.shape} -> (10 classes, 60000 amostras)")
    print(f"Shape X_test:  {X_test.shape}  -> (784 features, 10000 amostras)")
    print(f"Shape Y_test:  {Y_test.shape}  -> (10 classes, 10000 amostras)")