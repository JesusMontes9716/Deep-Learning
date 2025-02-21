import java.util.Random;

public class Perceptron {
    protected double[] weights;
    protected double bias;
    protected double learningRate;
    protected int epochs;

    public Perceptron(int inputSize, double learningRate, int epochs) {
        this.weights = new double[inputSize];
        this.bias = new Random().nextDouble();
        this.learningRate = learningRate;
        this.epochs = epochs;
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextDouble();
        }
    }

    protected double activate(double sum) {
        return 1 / (1 + Math.exp(-sum));
    }

    protected double weightedSum(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    public void train(double[][] inputs, double[] outputs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double sum = weightedSum(inputs[i]);
                double prediction = activate(sum);
                double error = outputs[i] - prediction;

                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputs[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

    public double predict(double[] inputs) {
        return activate(weightedSum(inputs));
    }
}
