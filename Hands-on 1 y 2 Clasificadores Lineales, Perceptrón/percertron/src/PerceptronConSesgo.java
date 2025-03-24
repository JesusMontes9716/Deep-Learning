// Clase PerceptronConSesgo que extiende la clase Perceptron y utiliza sesgo
public class PerceptronConSesgo extends Perceptron {

    // Constructor que inicializa el perceptrón con sesgo
    public PerceptronConSesgo(int inputSize, double learningRate, int epochs) {
        super(inputSize, learningRate, epochs); // Llama al constructor de la clase base Perceptron
    }

    // Método train sobrescrito para entrenar el perceptrón con sesgo
    @Override
    public void train(double[][] inputs, double[] outputs) {
        // Itera por el número de épocas (ciclos de entrenamiento)
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0; // Inicializa el error total para cada época
            System.out.println("Época: " + (epoch + 1)); // Imprime el número de la época actual

            // Itera sobre todas las muestras de entrenamiento
            for (int i = 0; i < inputs.length; i++) {
                // Calcula la suma ponderada de las entradas más el sesgo
                double sum = weightedSum(inputs[i]) + bias;
                double prediction = activate(sum); // Aplica la función de activación (sigmoide) para obtener la predicción
                double error = outputs[i] - prediction; // Calcula el error como la diferencia entre la salida esperada y la predicción
                totalError += Math.abs(error); // Acumula el error absoluto de esta muestra

                // Muestra los cálculos paso a paso para verificar el proceso
                System.out.printf("Entrada: %f, %f -> Suma: %f, Activación: %f, Error: %f\n",
                        inputs[i][0], inputs[i][1], sum, prediction, error);

                // Actualización de los pesos según el error, tasa de aprendizaje y las entradas
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputs[i][j]; // Actualiza los pesos de acuerdo al error
                }
                bias += learningRate * error; // Actualiza el sesgo de acuerdo al error
            }

            // Si el error total es suficientemente pequeño, el perceptrón ha convergido y podemos detener el entrenamiento
            if (totalError < 0.01) {
                System.out.println("El perceptrón con sesgo ha convergido."); // Informa que el entrenamiento ha convergido
                break; // Detiene el ciclo de entrenamiento
            }
        }

        // Imprime los parámetros finales (pesos y sesgo) después del entrenamiento
        printParameters();
    }
}
