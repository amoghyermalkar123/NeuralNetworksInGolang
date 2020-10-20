package neuron

import (
	"fmt"
	"math"
)

type vector []float64

type Neurons struct {
	Vectors []vector
}

func Sigmoid(x float64) float64 {
	return 1/1 + math.Exp(x)
}

func (n *Neurons) PrintNeurons() {
	fmt.Println("neurons:\n", n.Vectors)
}

func (neurons *Neurons) linearMatrixTransformation(wM []float64) []float64 {
	var inputToPerceptron []float64
	// vector multiplication with weights
	for i := 0; i < len(neurons.Vectors); i++ {
		for j := 0; j < len(neurons.Vectors[i]); j++ {
			neurons.Vectors[i][j] = neurons.Vectors[i][j] * wM[0]
		}
	}

	// add all vectors to prepare the final transformation
	for k := 0; k < len(neurons.Vectors)-1; k++ {
		for l := 0; l < len(neurons.Vectors[k]); l++ {
			temp := neurons.Vectors[k][l] + neurons.Vectors[k+1][l]
			inputToPerceptron = append(inputToPerceptron, temp)
		}
	}
	return inputToPerceptron
}

func perceptron(input []float64) (output []float64) {
	for item := range input {
		res := Sigmoid(input[item])
		output = append(output, res)
	}
	return output
}

// func buildInputVector() {

// }

func I() {
	// define input vectors
	inputMatrix := []vector{}
	weightMatrix := []float64{}

	// inits
	in1 := vector{1.0, 2.0}
	in2 := vector{2.0, 1.0}
	w1 := 0.01
	w2 := 0.01

	// matrix building
	inputMatrix = append(inputMatrix, in1)
	inputMatrix = append(inputMatrix, in2)
	neurons := &Neurons{}
	for i := 0; i < 2; i++ {
		neurons.Vectors = append(neurons.Vectors, inputMatrix[i])
	}
	weightMatrix = append(weightMatrix, w1, w2)

	// perform matrix multiplication
	inputToPerceptron := neurons.linearMatrixTransformation(weightMatrix)
	// pass result to perceptron
	outputVector := perceptron(inputToPerceptron)
	fmt.Println(outputVector)
	// cost calculation

	// weight calibration
}
