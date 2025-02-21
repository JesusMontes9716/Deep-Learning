public class Main {
    public static void main(String[] args) {
        double[][] andInputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[] andOutputs = {0, 0, 0, 1};

        double[][] orInputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[] orOutputs = {0, 1, 1, 1};

        PerceptronSinSesgo perceptronAnd = new PerceptronSinSesgo(2, 0.1, 1000);
        perceptronAnd.train(andInputs, andOutputs);
        System.out.println("Resultados Perceptrón SIN sesgo para AND:");
        for (double[] input : andInputs) {
            System.out.println("Entrada: " + input[0] + ", " + input[1] + " -> Salida: " + perceptronAnd.predict(input));
        }

        PerceptronConSesgo perceptronOr = new PerceptronConSesgo(2, 0.1, 1000);
        perceptronOr.train(orInputs, orOutputs);
        System.out.println("\nResultados Perceptrón CON sesgo para OR:");
        for (double[] input : orInputs) {
            System.out.println("Entrada: " + input[0] + ", " + input[1] + " -> Salida: " + perceptronOr.predict(input));
        }
    }
}
