# 🧠 Rede Neural do Zero - Classificação de Dígitos (MNIST)

Daniel Salvini - 2021101024

Este projeto implementa uma Rede Neural Artificial (Multi-Layer Perceptron) a partir do zero utilizando Python, tendo como objetivo a classificação de dígitos manuscritos da popular base de dados [MNIST](http://yann.lecun.com/exdb/mnist/).

Toda a arquitetura, propagação (forward/backward) e cálculos de treinamento foram estruturados de forma modular sem uso de frameworks de Deep Learning (como TensorFlow ou PyTorch) para fins de estudo e compreensão matemática do modelo.

## 📂 Estrutura do Projeto

```text
t1-redesneurais/
├── data/                       # Diretório com a base de dados do MNIST
│   ├── t10k-images.idx3-ubyte  # Imagens de teste (10.000 amostras)
│   ├── t10k-labels.idx1-ubyte  # Rótulos de teste
│   ├── train-images.idx3-ubyte # Imagens de treino (60.000 amostras)
│   └── train-labels.idx1-ubyte # Rótulos de treino
├── src/                        # Código-fonte e módulos do projeto
│   ├── data_preprocessing.py   # Normalização e preparação dos dados
│   ├── evaluation.py           # Métricas e avaliação do modelo
│   ├── mnist_loader.py         # Leitura dos arquivos binários IDX
│   ├── network.py              # Definição e inicialização de pesos e vieses
│   ├── propagation.py          # Lógica de Forward e Backward propagation
│   └── train.py                # Loop de treinamento (Épocas, Loss, Otimização)
├── main.py                     # Script principal que orquestra o pipeline
└── README.md                   # Documentação do projeto
```

## ⚙️ Pré-requisitos

Certifique-se de ter o Python 3.x instalado em sua máquina. Para gerenciar as dependências, recomenda-se criar um ambiente virtual (venv).

O projeto utiliza primariamente bibliotecas padrão do Python (como `os` e `struct` no loader) e a biblioteca essencial para cálculo matricial de alta performance:

* `numpy` (Para as operações matemáticas, produtos escalares e manipulação das matrizes de imagem)

Instale as dependências executando (caso crie um arquivo `requirements.txt`) ou instale manualmente:

```bash
pip install numpy
```

## 🚀 Como Executar o Projeto

**1. Clone ou baixe o repositório:**

```bash
git clone https://github.com/danielsalvini2002/t1-redes-neurais.git
cd t1-redes-neurais
```

**2. Verifique os dados:**
Certifique-se de que a pasta `data/` existe na raiz do projeto e contém os quatro arquivos originais (`.idx3-ubyte` e `.idx1-ubyte`) já descompactados.

**3. Execute o programa principal:**
Chame o script `main.py` direto da raiz do projeto:

```bash
python main.py
```

### O que acontece durante a execução?

O `main.py` roda o seguinte pipeline:

1. **Carregamento**: O dataset é lido, e as escalas de pixels (0-255) são normalizadas.
2. **Treinamento**: Inicia o laço de aprendizado da Rede Neural. Os hiperparâmetros padrão são:
   * **Épocas:** 30
   * **Taxa de Aprendizado (Learning Rate):** 0.1
   * **Tamanho do Lote (Batch Size):** 128
3. **Avaliação**: O modelo processará as imagens do conjunto de teste (10.000 imagens que nunca viu antes) e exibirá a Acurácia alcançada.

## 🎛️ Ajustando os Hiperparâmetros

Caso queira fazer testes e melhorar (ou piorar) o desempenho da rede, basta editar estas variáveis de configuração diretamente no arquivo `main.py` (linha 34):

```python
    EPOCHS = 30
    LEARNING_RATE = 0.1
    BATCH_SIZE = 128
```

Aumentar o número de épocas fará o programa rodar por mais tempo, mas possivelmente atingirá um nível de precisão maior.

## Notas do Desenvolvedor

Não consegui criar o código capaz de gerar os gráficos para melhor explorar os resultados atingidos, então é algo para fazer em futuras atualizaçôes. Mas fui capaz de percerber que a acurácia do modelo é de aproximadamente 92% no conjunto de teste para 30 epochs (menos que isso diminuiu drasticamente os resultados), o que é um resultado razoável para uma implementação do zero sem otimizações avançadas.

# 📚 Agradecimentos

* Obrigado pelo @AzorMesmo que me ajudou com a linguagem python ja que não tenho muita experiência com ela, e pelo professor GianCarlo que me auxiliou com o entendimento teórico em uma monitoria particular.
