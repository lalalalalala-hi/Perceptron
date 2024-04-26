import numpy as np

threshold = 0.2
learning_rate = 0.1

class Perceptron:
    @staticmethod
    def activation(out, threshold=0.2):
        return 1 if out >= threshold else 0
    
    @staticmethod
    def perceptron(inputs, outputs):
        x = np.array(inputs)
        y = np.array(outputs)
        w = np.array([0.3, -0.1])
        lr = learning_rate
        n = 5
        
        for _ in range(n):
            print("Epoch", _)
            print("x[i]\tw\t\tsumout\tout\tupdated w")

            for i in range(y.size):
                print(x[i], end='\t')
                print([round(val, 1) for val in w], end='\t')
                out = round(np.sum(w * x[i]), 2)
                print(out, end='\t')
                out = Perceptron.activation(out, threshold)
                print(out, end='\t')
                w += lr * (y[i] - out) * x[i]
                print([round(val, 1) for val in w], end='\t\n')
        return w
    
    @staticmethod
    def predict(inputs, weights):
        print(inputs, end='\t')
        print(weights, end='\t')
        out = round(np.sum(weights * inputs), 2)
        print(out, end='\t')
        return Perceptron.activation(out, threshold)
    
    @staticmethod
    def main():
        print("AND Gate:")
        weights = Perceptron.perceptron([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
        print("\nPredicting")
        print("x[i]\tw\t\tsumout\tout")
        for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print('-->', Perceptron.predict(inputs, weights))

        print("\n\n")

        print("OR Gate:")
        weights = Perceptron.perceptron([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1])
        print("\nPredicting")
        print("x[i]\tw\t\tsumout\tout")
        for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print('-->', Perceptron.predict(inputs, weights))


if __name__ == '__main__':
    Perceptron.main()
