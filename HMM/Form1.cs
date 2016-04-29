#define STRHC1
// STRHC1 = 10 gestures
// Not STRHC1 = 4 gestures

using Accord.Statistics.Distributions.Fitting;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Models.Markov;
using Accord.Statistics.Models.Markov.Learning;
using Accord.Statistics.Models.Markov.Topology;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HMM
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            //double[][][] sequences = new double[][][]
            //{
            //    new double[][] 
            //    { 
            //        // This is the first  sequence with label = 0 
            //        new double[] { 0, 1 },
            //        new double[] { 1, 2 },
            //        new double[] { 2, 3 },
            //        new double[] { 3, 4 },
            //        new double[] { 4, 5 },
            //    }, 

            //    new double[][]
            //    {
            //            // This is the second sequence with label = 1 
            //        new double[] { 4,  3 },
            //        new double[] { 3,  2 },
            //        new double[] { 2,  1 },
            //        new double[] { 1,  0 },
            //        new double[] { 0, -1 },
            //    }
            //};

            // Labels for the sequences 
            //int[] labels = { 0, 1 };

            #region Old

            //Dictionary<string, List<List<double[]>>> dictAll;
            //using (System.IO.FileStream fs = new System.IO.FileStream(@".\SkeletonsAsDouble.serialized", System.IO.FileMode.Open))
            //{
            //    BinaryFormatter bf = new BinaryFormatter();
            //    dictAll = (Dictionary<string, List<List<double[]>>>)bf.Deserialize(fs);
            //}

            //#region Export to CSV

            //using (System.IO.StreamWriter sw = new System.IO.StreamWriter(@".\SkeletonsAsCSV.csv"))
            //{
            //    double[] rgFirstFrame = dictAll[dictAll.Keys.ElementAt(0)][0][0];

            //    string strHeader = string.Empty;
            //    for (int i = 0; i < rgFirstFrame.Length; i++)
            //    {
            //        strHeader += string.Format("F{0},", i);
            //    }
            //    strHeader += "Label";
            //    sw.WriteLine(strHeader);

            //    foreach (KeyValuePair<string, List<List<double[]>>> kvp in dictAll)
            //    {
            //        foreach (List<double[]> lstGesture in kvp.Value)
            //        {
            //            foreach (double[] frame in lstGesture)
            //            {
            //                sw.WriteLine(string.Format("{0},{1}", string.Join(",", frame), kvp.Key));
            //            }
            //        }
            //    }
            //}

            #endregion

            #region CHANGE THESE PARAMETERS

            const int nStartHiddenCount = 2;
            const int nEndHiddenCount = 12;
            const int nFeatureCount = 8;    // Both DataOrig.dat and Data.dat have dimensionality of 8

            #endregion

            string strFile;
            int nClasses;

#if STRHC1
            // Data.dat = Second STRHC (10 classes but not as noisy)
            strFile = @".\Data.dat";
            nClasses = 10;
#else
            // Data1.dat = First STRHC (4 classes but very noisy)
            strFile = @".\DataOrig.dat";
            nClasses = 4;
#endif

            Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> dict;
            using (System.IO.FileStream fs = new System.IO.FileStream(strFile, System.IO.FileMode.Open, System.IO.FileAccess.Read))
            {
                BinaryFormatter bf = new BinaryFormatter();
                dict = bf.Deserialize(fs) as Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>>;
            }

            using (System.IO.StreamWriter sw = new System.IO.StreamWriter(@"HMMOutput.txt"))
            {
                for (int nHiddenCount = nStartHiddenCount; nHiddenCount <= nEndHiddenCount; nHiddenCount++)
                {
                    double fAverage = 0.0;

                    int nEpoch = 0;

                    List<int> lstTrainTimes = new List<int>();
                    List<int> lstRecogTimes = new List<int>();

                    //for (int i = 1; i <= nEpochs; i++)
                    foreach (KeyValuePair<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> kvpTT in dict)
                    {
                        nEpoch++;

                        #region Old

                        //Dictionary<string, List<List<double[]>>> dictTrain = new Dictionary<string, List<List<double[]>>>();
                        //Dictionary<string, List<List<double[]>>> dictTest = new Dictionary<string, List<List<double[]>>>();

                        //#region Divide the vectors into training and testing sets

                        //foreach (KeyValuePair<string, List<List<double[]>>> kvp in dictAll)
                        //{
                        //    int nGroupSize = kvp.Value.Count / nEpochs;

                        //    List<List<double[]>> lstTrain;
                        //    List<List<double[]>> lstTest;
                        //    dictTrain.Add(kvp.Key, (lstTrain = new List<List<double[]>>()));
                        //    dictTest.Add(kvp.Key, (lstTest = new List<List<double[]>>()));

                        //    for (int j = 0; j < kvp.Value.Count; j++)
                        //    {
                        //        if (j < i * nGroupSize && j >= (i * nGroupSize) - nGroupSize)
                        //        {
                        //            lstTest.Add(kvp.Value[j]);
                        //        }
                        //        else
                        //        {
                        //            lstTrain.Add(kvp.Value[j]);
                        //        }
                        //    }
                        //}

                        //#endregion

                        #endregion

                        Dictionary<string, List<List<double[]>>> dictTrain = kvpTT.Value.Item1;
                        Dictionary<string, List<List<double[]>>> dictTest = kvpTT.Value.Item2;

                        double[][][] sequences = new double[dictTrain.Sum(kvp => kvp.Value.Count)][][];
                        int[] labels = new int[sequences.Length];

                        #region The Sequences

                        int nIndex = 0;
                        foreach (KeyValuePair<string, List<List<double[]>>> kvp in dictTrain)
                        {
                            foreach (List<double[]> lst in kvp.Value)
                            {
                                sequences[nIndex] = lst.ToArray();
                                labels[nIndex] = Array.IndexOf(dictTrain.Keys.ToArray(), kvp.Key);
                                nIndex++;
                            }
                        }

                        #endregion

                        var initialDensity = new MultivariateNormalDistribution(nFeatureCount);

                        // Creates a sequence classifier containing 2 hidden Markov Models with 2 states 
                        // and an underlying multivariate mixture of Normal distributions as density. 
                        var classifier = new HiddenMarkovClassifier<MultivariateNormalDistribution>(
                            classes: nClasses, topology: new Forward(nHiddenCount), initial: initialDensity);

                        // Configure the learning algorithms to train the sequence classifier 
                        var teacher = new HiddenMarkovClassifierLearning<MultivariateNormalDistribution>(
                            classifier,

                            // Train each model until the log-likelihood changes less than 0.0001
                            modelIndex => new BaumWelchLearning<MultivariateNormalDistribution>(
                                classifier.Models[modelIndex])
                            {
                                Tolerance = 0.0001,
                                Iterations = 0,

                                FittingOptions = new NormalOptions()
                                {
                                    Diagonal = true,      // only diagonal covariance matrices
                                    Regularization = 1e-5 // avoid non-positive definite errors
                                }
                            }
                        );

                        // Train the sequence classifier using the algorithm 

                        System.Diagnostics.Stopwatch watchTrain = new System.Diagnostics.Stopwatch();
                        watchTrain.Start();
                        double logLikelihood = teacher.Run(sequences, labels);
                        watchTrain.Stop();

                        lstTrainTimes.Add((int)watchTrain.ElapsedMilliseconds);

                        //// Calculate the probability that the given 
                        ////  sequences originated from the model 
                        //double likelihood, likelihood2;

                        //// Try to classify the 1st sequence (output should be 0) 
                        //int c1 = classifier.Compute(sequences[0], out likelihood);

                        //// Try to classify the 2nd sequence (output should be 1) 
                        //int c2 = classifier.Compute(sequences[1], out likelihood2);

                        sw.WriteLine("Epoch: {0} -- Hidden State: {1}\t\tTotal Train Time: {2}", nEpoch, nHiddenCount, watchTrain.ElapsedMilliseconds.ToString());

                        int nCorrect = 0;
                        int nIncorrect = 0;

                        foreach (KeyValuePair<string, List<List<double[]>>> kvp in dictTest)
                        {
                            foreach (List<double[]> lst in kvp.Value)
                            {
                                System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();
                                watch.Start();
                                int nClassIndex = classifier.Compute(lst.ToArray());
                                watch.Stop();

                                sw.WriteLine(string.Format("Should be: {0}\tRecognized: {1}\t\tTime: {2} ms", kvp.Key, dictTest.Keys.ElementAt(nClassIndex), watch.ElapsedMilliseconds));
                                lstRecogTimes.Add((int)watch.ElapsedMilliseconds);

                                if (dictTest.Keys.ElementAt(nClassIndex) == kvp.Key)
                                {
                                    nCorrect++;
                                }
                                else
                                {
                                    nIncorrect++;
                                }
                            }
                        }

                        fAverage += (double)nCorrect / (nCorrect + nIncorrect);

                        sw.WriteLine(string.Format("Correct: {0} of {1} ({2:P3})", nCorrect, nCorrect + nIncorrect, (double)nCorrect / (nCorrect + nIncorrect)));

                        sw.WriteLine();
                        sw.WriteLine();
                    }

                    sw.WriteLine("Average Correct for {0} Hidden: {1:P3}", nHiddenCount, fAverage / 5);

                    sw.WriteLine("Average Train Time for {0}: {1:F2}", nHiddenCount, lstTrainTimes.Select(v => (double)v).Average());
                    sw.WriteLine("Std. Dev. Train Time for {0}: {1:F2}", nHiddenCount, lstTrainTimes.Select(v => (double)v).StandardDeviation());

                    sw.WriteLine("Average Recog. Time for {0}: {1:F2}", nHiddenCount, lstRecogTimes.Select(v => (double)v).Average());
                    sw.WriteLine("Std. Dev. Recog. Time for {0}: {1:F2}", nHiddenCount, lstRecogTimes.Select(v => (double)v).StandardDeviation());

                    sw.WriteLine();
                    sw.WriteLine();
                    sw.WriteLine();
                    sw.WriteLine();
                }
            }
        }
    }

    public static class Extensions
    {
        public static double StandardDeviation(this IEnumerable<double> values)
        {
            double avg = values.Average();
            return Math.Sqrt(values.Average(v => Math.Pow(v - avg, 2)));
        }
    }
}
