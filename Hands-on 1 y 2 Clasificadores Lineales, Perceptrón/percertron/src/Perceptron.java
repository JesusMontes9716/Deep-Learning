import java.util.Random; // Importa la clase Random para la generación de números aleatorios

// Clase Perceptron que implementa un perceptrón básico con sesión o sin sesión
public class Perceptron {
    protected double[] weights; // Array para almacenar los pesos del perceptrón
    protected double bias; // Sesgo (bias) del perceptrón
    protected double learningRate; // Tasa de aprendizaje que determina qué tan rápido se ajustan los pesos
    protected int epochs; // Número de épocas o ciclos de entrenamiento

    // Constructor que inicializa el perceptrón con el tamaño de entrada, la tasa de aprendizaje y el número de épocas
    public Perceptron(int inputSize, double learningRate, int epochs) {
        this.weights = new double[inputSize]; // Inicializa el array de pesos con el tamaño de las entradas
        this.bias = new Random().nextDouble(); // Inicializa el sesgo con un valor aleatorio entre 0 y 1
        this.learningRate = learningRate; // Asigna la tasa de aprendizaje
        this.epochs = epochs; // Asigna el número de épocas
        initializeWeights(); // Llama al método para inicializar los pesos
    }

    // Método para inicializar los pesos con valores aleatorios
    private void initializeWeights() {
        Random rand = new Random(); // Crea un objeto Random para generar números aleatorios
        for (int i = 0; i < weights.length; i++) { // Itera sobre el número de pesos
            weights[i] = rand.nextDouble(); // Asigna un valor aleatorio entre 0 y 1 a cada peso
        }
    }

    // Función de activación sigmoide
    protected double activate(double sum) {
        return 1 / (1 + Math.exp(-sum)); // Aplica la función sigmoide a la suma ponderada
    }

    // Método para calcular la suma ponderada de las entradas
    protected double weightedSum(double[] inputs) {
        double sum = bias; // Comienza con el valor del sesgo
        for (int i = 0; i < weights.length; i++) { // Itera sobre cada peso
            sum += weights[i] * inputs[i]; // Suma el producto de cada peso por la entrada correspondiente
        }
        return sum; // Devuelve la suma ponderada
    }

    // Método para entrenar el perceptrón con un conjunto de entradas y salidas esperadas
    public void train(double[][] inputs, double[] outputs) {
        // Itera por el número de épocas
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0; // Inicializa el error total en cero
            System.out.println("Época: " + (epoch + 1)); // Imprime el número de la época actual

            // Itera sobre cada muestra de entrenamiento
            for (int i = 0; i < inputs.length; i++) {
                double sum = weightedSum(inputs[i]); // Calcula la suma ponderada de las entradas
                double prediction = activate(sum); // Aplica la función de activación para obtener la predicción
                double error = outputs[i] - prediction; // Calcula el error como la diferencia entre la salida esperada y la predicción
                totalError += Math.abs(error); // Acumula el error absoluto de esta muestra

                // Muestra los cálculos paso a paso
                System.out.printf("Entrada: %f, %f -> Suma: %f, Activación: %f, Error: %f\n",
                        inputs[i][0], inputs[i][1], sum, prediction, error);

                // Actualiza los pesos y el sesgo usando el algoritmo de retropropagación
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputs[i][j]; // Actualiza el peso correspondiente
                }
                bias += learningRate * error; // Actualiza el sesgo
            }

            // Si el error total es suficientemente pequeño, el perceptrón ha convergido y podemos detener el entrenamiento
            if (totalError < 0.01) {
                System.out.println("El perceptrón ha convergido."); // Informa que el entrenamiento ha convergido
                break; // Detiene el ciclo de entrenamiento
            }
        }

        // Imprime los parámetros finales después del entrenamiento
        printParameters();
    }

    // Método para hacer una predicción con el perceptrón
    public double predict(double[] inputs) {
        return activate(weightedSum(inputs)); // Aplica la función de activación a la suma ponderada de las entradas
    }

    // Método para imprimir los parámetros finales (pesos y sesgo) del perceptrón
    public void printParameters() {
        System.out.println("\nPesos finales:"); // Imprime los pesos finales
        for (int i = 0; i < weights.length; i++) { // Itera sobre todos los pesos
            System.out.printf("w[%d] = %f\n", i, weights[i]); // Muestra el valor de cada peso
        }
        System.out.printf("Sesgo = %f\n", bias); // Muestra el valor final del sesgo
    }
}
