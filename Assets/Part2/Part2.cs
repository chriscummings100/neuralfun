using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class Part2 : MonoBehaviour
{
    public RawImage m_data_image;
    public Texture2D m_current_texture;
    public TMPro.TextMeshProUGUI m_data_label;
    public TMPro.TextMeshProUGUI m_result_label;

    public class Network
    {
        public int[] m_sizes;
        public double[][][] m_weights;
        public double[][] m_biases;

        public float m_train_progress;
        public float m_epoch_progress;
        public float m_last_accuracy;


        public void Load(byte[] data)
        {
            Debug.Log($"Loading network from {data.Length}B buffer");

            int pos = 0;
            int num_layers = BitConverter.ToInt32(data, pos);
            pos += 4;
            Debug.Log($"  num_layers: {num_layers}");

            m_sizes = new int[num_layers];
            for (int i = 0; i < num_layers; i++)
            {
                m_sizes[i] = BitConverter.ToInt32(data, 4 + i * 4);
                Debug.Log($"  layer {i}: {m_sizes[i]}");
                pos += 4;
            }

            m_weights = new double[num_layers][][];
            m_biases = new double[num_layers][];

            //setup dummy values for the input layer
            {
                int size = LayerSize(0);
                m_biases[0] = new double[size];
                m_weights[0] = new double[size][];
                for (int neuron = 0; neuron < size; neuron++)
                    m_weights[0][neuron] = new double[0];
            }

            //load biases for all neurons in all layers
            for (int layer = 1; layer < num_layers; layer++)
            {
                //neuron count is neuron count in RECEIVING layer
                int num_neurons = LayerSize(layer);

                //fill bias value for each neuron
                m_biases[layer] = new double[num_neurons];
                for (int bias = 0; bias < num_neurons; bias++)
                {
                    m_biases[layer][bias] = BitConverter.ToDouble(data, pos);
                    pos += 8;
                }
            }

            //load weights for all neurons in all layers
            for (int layer = 1; layer < num_layers; layer++)
            {
                //neuron count is neuron count in RECEIVING layer
                int num_neurons = LayerSize(layer);

                //weights per neuron is the number of neurons in the SENDING layer
                int num_weights_per_neuron = LayerSize(layer - 1);

                //allocate and fill weight array for each neuron
                m_weights[layer] = new double[num_neurons][];
                for (int neuron = 0; neuron < num_neurons; neuron++)
                {
                    m_weights[layer][neuron] = new double[num_weights_per_neuron];
                    for (int weight = 0; weight < num_weights_per_neuron; weight++)
                    {
                        m_weights[layer][neuron][weight] = BitConverter.ToDouble(data, pos);
                        pos += 8;
                    }
                }
            }

            if (pos != data.Length)
                Debug.LogError("Data not fully consumed");
        }

        public void Load(string zipfilename)
        {
            // Load the network from a zip file using .net compression library
            using (FileStream fs = new FileStream(zipfilename, FileMode.Open))
            {
                using (System.IO.Compression.ZipArchive archive = new System.IO.Compression.ZipArchive(fs))
                {
                    foreach (System.IO.Compression.ZipArchiveEntry entry in archive.Entries)
                    {
                        using (Stream stream = entry.Open())
                        {
                            byte[] data = new byte[entry.Length];
                            stream.Read(data, 0, data.Length);
                            Load(data);
                            break;
                        }
                    }
                }
            }
        }

        public double[] FeedForward(Dataset.Data data)
        {
            FeedForward(data, out _, out List<double[]> activations);
            return activations[activations.Count - 1];
        }

        public void FeedForward(
            Dataset.Data data,
            out List<double[]> weighted_inputs,
            out List<double[]> activations)
        {
            weighted_inputs = new List<double[]>();
            activations = new List<double[]>();

            double[] a = data.m_inputs;

            //add activations of input layer, and a null set of weighted
            //inputs, as it had none!
            weighted_inputs.Add(null);
            activations.Add(a);

            //feed forward through the network, outputting weighted
            //inputs and activations at each step
            for (int layer = 1; layer < m_sizes.Length; layer++)
            {
                double[] weighted_input = new double[m_sizes[layer]];
                double[] biases = m_biases[layer];
                double[][] weights = m_weights[layer];
                for (int neuron = 0; neuron < m_sizes[layer]; neuron++)
                {
                    weighted_input[neuron] = biases[neuron] + Dot(weights[neuron], a);
                }

                a = Sigmoid(weighted_input);

                weighted_inputs.Add(weighted_input);
                activations.Add(a);
            }
        }

        int NumLayers => m_sizes.Length;

        int LayerSize(int layer_idx) => m_sizes[layer_idx];

        //classic array shuffling algorithm to get randomly shuffled version of the training data from a dataset
        Dataset.Data[] GetShuffledTrainingData(Dataset dataset)
        {
            //take copy of training data
            var in_data = new Dataset.Data[dataset.m_training_data.Length];
            for (int i = 0; i < in_data.Length; i++)
                in_data[i] = dataset.m_training_data[i];

            //randomly remove elements from the copy and apparent them to the output array
            var out_data = new Dataset.Data[dataset.m_training_data.Length];
            int end = in_data.Length;
            int write = 0;
            while (end > 0)
            {
                int idx = UnityEngine.Random.Range(0, end);
                out_data[write++] = in_data[idx];
                in_data[idx] = in_data[--end];
            }

            if (write != out_data.Length)
                throw new Exception("Shuffle failed");
            return out_data;
        }

        public IEnumerator DoTrain(Dataset dataset, int epochs)
        {
            m_train_progress = 0;
            for (int i = 0; i < epochs; i++)
            {
                yield return DoTrainEpoch(dataset);
                m_train_progress = (i + 1) / (float) (epochs);
            }
        }

        public void Evaluate(Dataset dataset)
        {
            int count = 0;
            foreach (var data in dataset.m_test_data)
            {
                var activations = FeedForward(data);
                int max_index = MaxIndex(activations);
                if (max_index == data.m_output)
                    count++;
            }

            m_last_accuracy = count / (float) dataset.m_test_data.Length;
            Debug.Log("Accuracy: " + m_last_accuracy);
        }

        public IEnumerator DoTrainEpoch(Dataset dataset)
        {
            m_epoch_progress = 0;

            //get randomly shuffled training data
            var training_data = GetShuffledTrainingData(dataset);

            //divide into batches, and train network off each batch 1 at a time
            int first_idx = 0;
            int batch_size = 10;
            while (first_idx < training_data.Length)
            {
                var batch = new ArraySegment<Dataset.Data>(
                    training_data, first_idx, batch_size);
                TrainBatch(batch);
                first_idx += batch_size;
                m_epoch_progress = first_idx / (float) training_data.Length;
                yield return null;
            }

            //evaluate network after all batches done
            Evaluate(dataset);
        }

        //randomly initialize weights and biases (set to random values between -1 and 1)
        public void Randomize()
        {
            for (int layer_idx = 0; layer_idx < NumLayers; layer_idx++)
            {
                for (int neuron = 0; neuron < LayerSize(layer_idx); neuron++)
                {
                    m_biases[layer_idx][neuron] = UnityEngine.Random.Range(-1f, 1f);
                    for (int weight_idx = 0; weight_idx < m_weights[layer_idx][neuron].Length; weight_idx++)
                    {
                        m_weights[layer_idx][neuron][weight_idx] = UnityEngine.Random.Range(-1f, 1f);
                    }
                }
            }
        }

        //train the network on a batch of data
        public void TrainBatch(ArraySegment<Dataset.Data> batch)
        {
            //these are gradients of the cost function with respect to the weights and biases
            //they all start out by 0, and are adjusted by the backpropagation algorithm
            var dweight_by_dcost = new double[NumLayers][][];
            var dbias_by_dcost = new double[NumLayers][];
            for (int layer_idx = 1; layer_idx < NumLayers; layer_idx++)
            {
                int layersize = LayerSize(layer_idx);
                dbias_by_dcost[layer_idx] = new double[layersize];
                dweight_by_dcost[layer_idx] = new double[layersize][];
                int numweights = LayerSize(layer_idx - 1);
                for (int neuron = 0; neuron < layersize; neuron++)
                    dweight_by_dcost[layer_idx][neuron] = new double[numweights];
            }

            foreach (var datapoint in batch)
            {
                //run back prop for this data point, which gives us
                //a dweight_by_cost and dbias_by_dcost just for this data point
                CalculateBackPropGradients(datapoint,
                    out double[][][] datapoint_dweight_by_dcost,
                    out double[][] datapoint_dbias_by_dcost);

                //now add the gradients for this data point to the running total
                for (int layer_idx = 1; layer_idx < NumLayers; layer_idx++)
                {
                    for (int neuron = 0; neuron < LayerSize(layer_idx); neuron++)
                    {
                        dbias_by_dcost[layer_idx][neuron] +=
                            datapoint_dbias_by_dcost[layer_idx][neuron];

                        for (int weight_idx = 0;
                            weight_idx < dweight_by_dcost[layer_idx][neuron].Length;
                            weight_idx++)
                        {
                            dweight_by_dcost[layer_idx][neuron][weight_idx] +=
                                datapoint_dweight_by_dcost[layer_idx][neuron][weight_idx];
                        }
                    }
                }
            }

            //finally, we can apply 'gradient descent' to the
            //weights and biases - i.e. adjust them in the 
            //opposite direction to the gradient, scaled
            //by constant learning rate over batch size
            {
                double batch_learning_rate = 3.0 / (double) batch.Count;

                //now add the gradients for this data point to the running total
                for (int layer_idx = 1; layer_idx < NumLayers; layer_idx++)
                {
                    for (int neuron = 0; neuron < LayerSize(layer_idx); neuron++)
                    {
                        m_biases[layer_idx][neuron] -=
                            batch_learning_rate * dbias_by_dcost[layer_idx][neuron];

                        for (int weight_idx = 0;
                            weight_idx < m_weights[layer_idx][neuron].Length;
                            weight_idx++)
                        {
                            m_weights[layer_idx][neuron][weight_idx] -=
                                batch_learning_rate * dweight_by_dcost[layer_idx][neuron][weight_idx];
                        }
                    }
                }
            }
        }

        public void CalculateBackPropGradients(Dataset.Data data, out double[][][] dweight_by_dcost, out double[][] dbias_by_dcost)
        {
            //these are gradients of the cost function with respect to the weights and biases
            dweight_by_dcost = new double[NumLayers][][];
            dbias_by_dcost = new double[NumLayers][];

            //do feed forward step and record all zs and activations
            FeedForward(data, out List<double[]> weighted_inputs, out List<double[]> activations);

            int layer_idx = NumLayers - 1;
            double[] weighted_input_gradient;

//calculate weight input error and bias/weight gradients for the output layer
            {
                //calculate the error in the output vs the target from the data (expressed
                //as 10 element vector with 1 for the correct digit) mathematically this is
                //calculated the partial derivative of the cost function with respect to
                //the output of the last layer
                double[] activation_gradient = Sub(activations[layer_idx],
                    data.m_vectorized_output);

                //convert the output error to the input error for the last layer by
                //applying the sigmoid derivative. In effect, this is taking the output
                //error and scaling it according to the slope of the sigmoid function
                //at the weighted input
                weighted_input_gradient = Mul(activation_gradient,
                    SigmoidDerivative(weighted_inputs[layer_idx]));

                //bias gradient is equal to the weighted input error 
                dbias_by_dcost[layer_idx] = weighted_input_gradient;

                //each weight gradient for each neuron is the activation of the corresponding
                //connected neuron in the previous layer times the weighted input error
                dweight_by_dcost[layer_idx] = new double[LayerSize(layer_idx)][];
                for (int neuron = 0; neuron < LayerSize(layer_idx); neuron++)
                {
                    dweight_by_dcost[layer_idx][neuron] =
                        Mul(activations[layer_idx - 1], weighted_input_gradient[neuron]);
                }

                //step to next layer
                layer_idx--;
            }

//back prop through layers (note: should really stop at layer 1, but I'm intrigued by hypothetical
//bias values for the input layer)
            while (layer_idx >= 1)
            {
                //this is the most special bit! here we're taking the weighted input error from the next layer and
                //using it to calculate the weighted input error for this layer. aka back propagation!
                {
                    double[] next_weighted_input_error = new double[LayerSize(layer_idx)];

                    for (int neuron = 0; neuron < LayerSize(layer_idx); neuron++)
                    {
                        //mathematically, m_weights[layer_idx+1] is a matrix, and what we're really doing here is
                        //extracting the column for this neuron, and dotting it with the weighted input error.
                        //done as 1 big mathematical operation for all neurons at once, this equivalent to taking
                        //the transpose of the weight matrix and multiplying it by the weighted input error,
                        //hence why NN books talk about weight matrix transposes and stuff!

                        //calculated weighted sum of the weighted input gradients we've
                        //already calculated for the connected neurons
                        double next_activation_gradient = 0;
                        for (int recv_neuron = 0;
                            recv_neuron < LayerSize(layer_idx + 1);
                            recv_neuron++)
                        {
                            next_activation_gradient +=
                                m_weights[layer_idx + 1][recv_neuron][neuron] *
                                weighted_input_gradient[recv_neuron];
                        }

                        //with the sum, we can now calculate the weighted input
                        //error in this layer for this neuron
                        next_weighted_input_error[neuron] =
                            next_activation_gradient *
                            SigmoidDerivative(weighted_inputs[layer_idx])[neuron];
                    }

                    weighted_input_gradient = next_weighted_input_error;
                }

                //repeat the process of calculating the bias and weight gradients
                //for this layer just as we did for the first layer
                dbias_by_dcost[layer_idx] = weighted_input_gradient;
                dweight_by_dcost[layer_idx] = new double[LayerSize(layer_idx)][];
                for (int neuron = 0; neuron < LayerSize(layer_idx); neuron++)
                {
                    dweight_by_dcost[layer_idx][neuron] =
                        Mul(activations[layer_idx - 1], weighted_input_gradient[neuron]);
                }

                layer_idx--;
            }

        }


    }

    Network m_network;
    Dataset m_dataset;
    int m_current_data = 0;
    
    public ProgressBar m_training_progress_bar;
    public ProgressBar m_epoch_progress_bar;
    public ProgressBar m_accuracy_bar;

    // Start is called before the first frame update
    void Start()
    {
        m_network = new ();
        m_network.Load(Application.streamingAssetsPath + "/network.zip");
        
        m_dataset = new ();
        m_dataset.Load(Application.streamingAssetsPath + "/mnist.zip");

        m_current_data = 0;
        SetData(m_dataset.m_training_data[0]);
    }

    void SetData(Dataset.Data data)
    {
        if(m_current_texture)
            Destroy(m_current_texture);
        m_current_texture = data.CreateTexture();
        m_data_image.texture = m_current_texture;
        m_data_label.text = data.m_output.ToString();

        double[] results = m_network.FeedForward(data);
        m_result_label.text = MaxIndex(results).ToString();
    }

    public void OnPrevClicked()
    {
        m_current_data = (m_current_data+m_dataset.m_training_data.Length-1) % m_dataset.m_training_data.Length;
        SetData(m_dataset.m_training_data[m_current_data]);
    }

    public void OnNextClicked()
    {
        m_current_data = (m_current_data+1) % m_dataset.m_training_data.Length;
        SetData(m_dataset.m_training_data[m_current_data]);
    }

    public void OnEvaluateClicked()
    {
        m_network.Evaluate(m_dataset);
    }
    public void OnRandomizeClicked()
    {
        m_network.Randomize();
    }
    public void OnTrainEpockClicked()
    {
        StartCoroutine(m_network.DoTrainEpoch(m_dataset));
    }
    
    public void OnTrainClicked()
    {
        StartCoroutine(m_network.DoTrain(m_dataset,30));
    }

    void OnDestroy()
    {
        if(m_current_texture)
            Destroy(m_current_texture);
    }

    // Update is called once per frame
    void Update()
    {
        if (m_network != null)
        {
            if(m_training_progress_bar)
                m_training_progress_bar.m_progress = m_network.m_train_progress;
            if(m_epoch_progress_bar)
                m_epoch_progress_bar.m_progress = m_network.m_epoch_progress;
            if (m_accuracy_bar)
                m_accuracy_bar.m_progress = m_network.m_last_accuracy;
        }
        
    }
    
    public static void InverseSigmoid(double[] z, double[] res)
    {
        for (int i = 0; i < z.Length; i++)
            res[i] = InverseSigmoid(z[i]);
    }

    public static double[] InverseSigmoid(double[] z)
    {
        var res = new double[z.Length];
        for (int i = 0; i < z.Length; i++)
            res[i] = InverseSigmoid(z[i]);
        return res;
    }

    public static double InverseSigmoid(double z)
    {
        return Math.Log(z/(1 - z));
    }
    
    public static void Sigmoid(double[] z, double[] res)
    {
        for (int i = 0; i < z.Length; i++)
            res[i] = Sigmoid(z[i]);
    }

    public static double[] Sigmoid(double[] z)
    {
        var res = new double[z.Length];
        for (int i = 0; i < z.Length; i++)
            res[i] = Sigmoid(z[i]);
        return res;
    }
 
    public static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    public static double SigmoidDerivative(double z)
    {
        return Sigmoid(z) * (1 - Sigmoid(z));
    }
    
    public static void SigmoidDerivative(double[] z, double[] res)
    {
        for (int i = 0; i < z.Length; i++)
            res[i] = SigmoidDerivative(z[i]);
    }

    public static double[] SigmoidDerivative(double[] z)
    {
        var res = new double[z.Length];
        SigmoidDerivative(z, res);
        return res;
    }

    public static void Add(double[] a, double[] b, double[] res)
    {
        for (int i = 0; i < a.Length; i++)
            res[i] = a[i]+b[i];
    }

    public static double[] Add(double[] a, double[] b)
    {
        var res = new double[a.Length];
        Add(a, b, res);
        return res;
    }
    
    public static void Mul(double[] a, double[] b, double[] res)
    {
        for (int i = 0; i < a.Length; i++)
            res[i] = a[i]*b[i];
    }

    public static double[] Mul(double[] a, double[] b)
    {
        var res = new double[a.Length];
        Mul(a, b, res);
        return res;
    }

    public static void Mul(double[] a, double b, double[] res)
    {
        for (int i = 0; i < a.Length; i++)
            res[i] = a[i]*b;
    }

    public static double[] Mul(double[] a, double b)
    {
        var res = new double[a.Length];
        Mul(a, b, res);
        return res;
    }

    public static double Dot(double[] a, double[] b)
    {
        double res = 0;
        for (int i = 0; i < a.Length; i++)
            res += a[i]*b[i];
        return res;
    }
    
    public static void Sub(double[] a, double[] b, double[] res)
    {
        for (int i = 0; i < a.Length; i++)
            res[i] = a[i]-b[i];
    }

    public static double[] Sub(double[] a, double[] b)
    {
        var res = new double[a.Length];
        Sub(a, b, res);
        return res;
    }

    public static int MaxIndex(double[] a)
    {
        double max_val = -double.MaxValue;
        int max_idx = -1;
        for(int i = 0; i < a.Length; i++)
        {
            if (a[i] > max_val)
            {
                max_val = a[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

}
