# Weather Prediction Analysis
Physicists define climate as a **“complex system”**. While there are a lot of interpretations about it, in this specific case we can consider **“complex”** to be **“unsolvable in analytical ways”**.
This may seems discouraging, but it actually paves the way to a wide range of numerical algorithms that aim to solve the climate challenges. With the computational developments of the last years, Machine Learning algorithms are certainly part of them.

![weather](https://user-images.githubusercontent.com/66861243/139519761-687fdb0c-44b6-4397-ab3d-a50ca2d779c9.png)

## Abstract
We have explored four Machine Learning algorithms namely Artificial Neural Networks, Gated Recurrent Units, Support Vector Regressor and Random Forests to study the weather patterns in the Chennai region of Tamil Nadu, India for the last ten years and use them for weather prediction. We have performed a comparative analysis among the aforementioned algorithms based on their performance in weather forecasting by using Mean Absolute Error(L1 Loss) and Mean Squared Error(L2 Loss) as evaluation metrics. We have conducted this research by collecting data from the Chennai Airport weather station. 

## Module Description
Model | Description | 
:-------------: | :---------: |
Artificial Neural Networks | An artificial neural network is an attempt to simulate the network of neurons that make up a human brain so that the computer will be able to learn things and make decisions in a humanlike manner. ANNs are created by programming regular computers to behave as though they are interconnected brain cells. | 
Support Vector Machine | Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. The basic idea behind SVR is to find the best fit line. In SVR, the best fit line is the hyperplane that has the maximum number of points. | 
Random Forest Regressor | The random forest combines hundreds or thousands of decision trees, trains each one on a slightly different set of the observations, splitting nodes in each tree considering a limited number of the features. The final predictions of the random forest are made by averaging the predictions of each individual tree. |
Gated Recurrent Units | Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks, introduced in 2014. The GRU is like a long short-term memory (LSTM) with a forget gate, but has fewer parameters than LSTM, as it lacks an output gate. GRU's performance on certain tasks of polyphonic music modeling, speech signal modeling and natural language processing was found to be similar to that of LSTM. |

## Quick Start
To use the repo and run inferences, please follow the guidelines below

- Cloning the Repository: 

        $ git clone https://github.com/indiradutta/Weather-Prediction-Analysis
        
- Entering the directory: 

        $ cd Weather-Prediction-Analysis/
        
- Setting up the Python Environment with dependencies:

        $ pip install -r requirements.txt

- Running the file for inference:

        $ streamlit run main.py
