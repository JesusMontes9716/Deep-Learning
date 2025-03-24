// Clase principal que contiene el método main para ejecutar el programa
public class Main {
    public static void main(String[] args) {

        // Matriz de entrada para la compuerta lógica AND (valores de entrada)
        double[][] andInputs = {
                {0, 0},  // Entrada 1
                {0, 1},  // Entrada 2
                {1, 0},  // Entrada 3
                {1, 1}   // Entrada 4
        };

        // Salidas esperadas para la compuerta lógica AND
        double[] andOutputs = {0, 0, 0, 1}; // Solo 1 cuando ambas entradas son 1

        // Matriz de entrada para la compuerta lógica OR (valores de entrada)
        double[][] orInputs = {
                {0, 0},  // Entrada 1
                {0, 1},  // Entrada 2
                {1, 0},  // Entrada 3
                {1, 1}   // Entrada 4
        };

        // Salidas esperadas para la compuerta lógica OR
        double[] orOutputs = {0, 1, 1, 1}; // 1 cuando al menos una entrada es 1

        // Imprime un mensaje indicando el inicio del entrenamiento para la compuerta AND
        System.out.println("Entrenando Perceptrón SIN sesgo para AND...");

        // Crea una instancia del Perceptrón SIN sesgo para la compuerta AND
        PerceptronSinSesgo perceptronAnd = new PerceptronSinSesgo(2, 0.1, 1000);

        // Entrena el perceptrón con los datos de la compuerta AND
        perceptronAnd.train(andInputs, andOutputs);

        // Imprime un mensaje indicando el inicio del entrenamiento para la compuerta OR
        System.out.println("\nEntrenando Perceptrón CON sesgo para OR...");

        // Crea una instancia del Perceptrón CON sesgo para la compuerta OR
        PerceptronConSesgo perceptronOr = new PerceptronConSesgo(2, 0.1, 1000);

        // Entrena el perceptrón con los datos de la compuerta OR
        perceptronOr.train(orInputs, orOutputs);

        // Imprime un mensaje indicando que se mostrarán los resultados finales
        System.out.println("\nResultados finales:");

        // Itera sobre cada entrada de la compuerta AND y predice su salida
        for (double[] input : andInputs) {
            System.out.printf("AND Entrada: %f, %f -> Salida: %f\n",
                    input[0], input[1], perceptronAnd.predict(input));
        }

        // Itera sobre cada entrada de la compuerta OR y predice su salida
        for (double[] input : orInputs) {
            System.out.printf("OR Entrada: %f, %f -> Salida: %f\n",
                    input[0], input[1], perceptronOr.predict(input));
        }
    }
}
