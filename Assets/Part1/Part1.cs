using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class Part1 : MonoBehaviour
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
        
        public void Load(byte[] data)
        {
            Debug.Log($"Loading network from {data.Length}B buffer");
            
            int pos = 0;
            int num_layers = BitConverter.ToInt32(data, pos); pos += 4;
            Debug.Log($"  num_layers: {num_layers}");
            
            m_sizes = new int[num_layers];
            for (int i = 0; i < num_layers; i++)
            {
                m_sizes[i] = BitConverter.ToInt32(data, 4 + i * 4);
                Debug.Log($"  layer {i}: {m_sizes[i]}");
                pos += 4;
            }
            
            m_weights = new double[num_layers - 1][][];
            m_biases = new double[num_layers - 1][];
           
            for (int layer = 0; layer < num_layers - 1; layer++)
            {
                //neuron count is neuron count in RECEIVING layer
                int num_neurons = m_sizes[layer + 1];
                
                //fill bias value for each neuron
                m_biases[layer] = new double[num_neurons];
                for (int bias = 0; bias < num_neurons; bias++)
                {
                    m_biases[layer][bias] = BitConverter.ToDouble(data, pos);
                    pos += 8;
                }
            }
         
            for (int layer = 0; layer < num_layers - 1; layer++)
            {
                //neuron count is neuron count in RECEIVING layer
                int num_neurons = m_sizes[layer + 1];
                
                //weights per neuron is the number of neurons in the SENDING layer
                int num_weights_per_neuron = m_sizes[layer];
                
                //allocate and fill weight array for each neuron
                m_weights[layer] = new double[num_neurons][];
                for (int neuron = 0; neuron < num_neurons; neuron++)
                {
                    m_weights[layer][neuron] = new double[num_weights_per_neuron];
                    for(int weight = 0; weight < num_weights_per_neuron; weight++)
                    {
                        m_weights[layer][neuron][weight] = BitConverter.ToDouble(data, pos);
                        pos += 8;
                    }
                }
            }
                
            if(pos != data.Length)
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
            //start with activations from the inputs
            double[] current_activations = data.m_inputs;
            
            //feed forward through the network
            for (int layer = 0; layer < m_sizes.Length - 1; layer++)
            {
                //get biases and weights for this layer
                double[] biases = m_biases[layer];
                double[][] weights = m_weights[layer];
                
                //calculate the new activations
                double[] new_activations = new double[m_sizes[layer + 1]];
                for (int neuron = 0; neuron < m_sizes[layer + 1]; neuron++)
                    new_activations[neuron] = Sigmoid(biases[neuron] + Dot(weights[neuron], current_activations));
                
                //store them
                current_activations = new_activations;
            }

            return current_activations;
        }
        

    }

    Network m_network;
    Dataset m_dataset;
    int m_current_data = 0;
    
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
    
    void OnDestroy()
    {
        if(m_current_texture)
            Destroy(m_current_texture);
    }

    public static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    public static double Dot(double[] a, double[] b)
    {
        double res = 0;
        for (int i = 0; i < a.Length; i++)
            res += a[i]*b[i];
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
