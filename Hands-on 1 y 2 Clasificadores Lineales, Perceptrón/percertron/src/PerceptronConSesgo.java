public class PerceptronConSesgo extends Perceptron {
    public PerceptronConSesgo(int inputSize, double learningRate, int epochs) {
        super(inputSize, learningRate, epochs);
    }

    @Override
    public void train(double[][] inputs, double[] outputs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double sum = weightedSum(inputs[i]) + bias;
                double prediction = activate(sum);
                double error = outputs[i] - prediction;

                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputs[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

    @Override
    public double predict(double[] inputs) {
        return activate(weightedSum(inputs) + bias);
    }
}
