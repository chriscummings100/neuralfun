using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

    public class Dataset
    {
        public Data[] m_training_data;
        public Data[] m_validation_data;
        public Data[] m_test_data;

        public class Data
        {
            //data from MNIST dataset
            public double[] m_inputs;
            public byte m_output;
            
            //10-element vector with 1.0 in the position of the output label
            public double[] m_vectorized_output;

            public Texture2D CreateTexture()
            {
                var tex = new Texture2D(28, 28, TextureFormat.RGB24, false);
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        byte v = (byte)(m_inputs[y * 28 + x] * 255);
                        tex.SetPixel(x, 27-y, new Color32(v, v, v, 255));
                    }
                }
                tex.Apply();
                return tex;
            }
        }
        

        Data LoadData(byte[] data, ref int pos)
        {
            Data d = new Data();
            d.m_inputs = new double[784];
            for (int i = 0; i < 784; i++)
            {
                d.m_inputs[i] = BitConverter.ToSingle(data, pos);
                pos += 4;
            }
            d.m_output = data[pos];
            d.m_vectorized_output = new double[10];
            d.m_vectorized_output[d.m_output] = 1.0;
            pos++;
            return d;
        }
        
        public void Load(byte[] data)
        {
            Debug.Log($"Loading dataset from {data.Length}B buffer");

            int pos = 0;

            {
                int data_len = BitConverter.ToInt32(data, pos);
                pos += 4;
                m_training_data = new Data[data_len];
                for (int i = 0; i < data_len; i++)
                    m_training_data[i] = LoadData(data, ref pos);
            }
            {
                int data_len = BitConverter.ToInt32(data, pos);
                pos += 4;
                m_validation_data = new Data[data_len];
                for (int i = 0; i < data_len; i++)
                    m_validation_data[i] = LoadData(data, ref pos);
            }
            {
                int data_len = BitConverter.ToInt32(data, pos);
                pos += 4;
                m_test_data = new Data[data_len];
                for (int i = 0; i < data_len; i++)
                    m_test_data[i] = LoadData(data, ref pos);
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
    }
    