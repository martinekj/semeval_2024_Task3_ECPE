# SemEval-2024 Task 3: The Competition of Multimodal Emotion Cause Analysis in Conversations
## Submission
### Team name: UWBA 
University of West Bohemia

jimar@kiv.zcu.cz


### Basic information about our approach
``bin`` folder contains model weights and arguments

``data`` folder contains temporary files, *audio* and *video* vectors for **eval data only**

In order to maintain the maximum level of clarity, our final model loads the previously-created audio and video features
in the form of ``csv`` files: **data/video_vectors_eval.csv** and **data/audio_vectors_eval_mfcc2k.tsv**

*Audio vectors*
- Audio ferature vectors are created with librosa library and its mfcc feature extraction method.
Values from this method are used as representation of the audio file. 
- These vectors are then fed into LSTM model. The model consists of one LSTM layer and two 
fully connected neural network layers. Model is than trained for emotion recognition task from provided
training data.
- Audio feature vectors are the last hidden state of LSTM layer provided by trained model.
Each vector has a dimension of 2048 and represents one audio file.
- Audio features are in the *csv* file: ``audio_vectors_eval_mfcc2k.tsv``

*Video vectors*
- The core of the visual/video encoder is  ResNexT-101 https://github.com/kenshohara/3D-ResNets-PyTorch
- Video features are extracted and 2048-dim feature vectors for each utterance are produced
- Video features are in the *csv* file: ``video_vectors_eval.csv`` 

 
*Text span approach*

We utilized ``bert-large-cased`` model for predicting text spans. It is a very simple approach where we defined
5 text span categories as follows:
- **first part**
- **middle part**
- **last part**
- **whole utterance**
- **other**

We use regular expreesions to find out which text span belong to a certain class. For details see ``text_spans_classifier``
folder. The model is able to predict the categories and we use the same regular expressions to extract text spans.
In ``text_spans_classifier`` folder, there is a code for creating such a model with.
Within our approach, we use only the file ``predicted_text_spans_eval_data.tsv`` which contains previously extracted text spans.
If the predicted class is **other** we use whole utterance as a text span. 

### How to run our model and reproduce results?

1. Set correct paths in ``config.py``

2. Run ``final_evaluation.py``
    

      The script first converts the format of ``Subtask_2_test.json`` to our format necessary for our models.
      Then model weights are loaded from checkpoints (two models: 1. predicting emotions, 2. predicting links).
      Predicting converted input json file and producing final ``Subtask_2_pred.json``.
   
   
      In the second phase, the script takes ``Subtask_2_pred.json`` file and create the copy of this file with text spans
      according to the required format and produces ``Subtask_1_pred.json``.

3. See the resulting files ``Subtask_1_pred.json`` and ``Subtask_2_pred.json``



### Data
Data available at SemEval 2024 Task 3 page: https://nustm.github.io/SemEval-2024_ECAC/
