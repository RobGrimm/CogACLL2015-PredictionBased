Code for obtaining results described in the following paper: 

> Grimm R., Cassani, G., Daelemans W., & Gillis S.. (2015). Towards a Model of Prediction-based Syntactic Category Acquisition: First Steps with Word Embeddings. Sixth Workshop on Cognitive Aspects of Computational Language Learning (CogACLL 2015), 28—32

Link: http://www.emnlp2015.org/proceedings/CogACLL/pdf/CogACLL05.pdf



## Dependencies

The system is implemented in Python 2.7.6  
The following libraries need to be installed (version number used during development is given in parentheses):  
> matplotlib (1.4.3)  
numpy (1.9.2)  
scipy (0.15.1)  
scikit-learn (0.16.1)  
nltk (3.0.3)  
Theano (0.7.0)


## Preparing the Data 

#### Prepare the CHILDES corpora 

Get the North American corpora from: http://childes.psy.cmu.edu/data-xml/Eng-NA-MOR/  
Then unzip them to: pred_syntactic/CHILDES/Corpora/ENG-NA/

Similarly, get the British English corpora from: http://childes.psy.cmu.edu/data-xml/Eng-UK-MOR/  
Unzip them to: pred_syntactic/CHILDES/Corpora/ENG-UK/

Download the following North American corpora:
> Bates, Bliss, Bloom73, Bohannon, Brown, Demetras1, Demetras2, Feldman, Hall, Kuczaj, MacWhinney, NewEngland, Suppes, Tardif,
Valian, VanKleeck

And the following British English corpora:  
> Belfast, Manchester

Once you have downloaded and unzipped the corpora to their folders, run pred_syntactic/CHILDES/extract.py   
This will read and process the corpus data, then pickle it to a dictionary for further use. 


#### Prepare the phonological data 

Phonological features are based on phonemic data from the CELEX database (http://wwwlands2.let.kun.nl/members/software/celex.html -- the CD-ROM release 2 was used during development, but it will probably also work with release 1).

You need to buy a license to be able to use CELEX legally. If you do not have access to the database, you can still run the experiments that do not depend on phonological features.

From the CELEX CD-ROM, copy the english\epw folder (contains phonological information for English words) to pred_syntactic\Phonology\Celex.
Then run pred_syntactic\Phonology\Celex\epw_to_dictionary.py, which will pickle phonological information to a dictionary for further use.


## Running the experiments 

First, run pred_syntactic\extract_text_data_set.py  
This will go through the training corpus, extract left-context vectors and right-context words, then save the data to disk. 

Next, run pred_syntactic\train_bimodal_auto_encoder.py  
This will train an Auto Encoder on (1) left-context vectors, (2) left-context vectors concatenated with phonological feature vectors, and (3) left-context vectors concatenated with lexical stress feature vectors. Dimensionality-reduced input vectors obtained from the trained Auto Encoder are saved to disk.

Finally, run pred_syntactic\train_softmax_model.py  
This will train a Softmax model to predict words from the right context. It will modify the dimensionality-reduced vectors from the
previous step to maximize the probability of each right-context word given the left context. 

## Results

Results are saved to pred_syntactic\results. Precision, recall, micro and macro F1 scores at each training stage are written to pred_syntactic\results\results_over_training_stages. A number of additional metrics are stored (plots of training error over epochs and micro F1 over epochs; T-SNE plots of word embeddings represented as dots colored by syntactic type; T-SNE plots of word embeddings represented as orthographic word forms). 

To compute statistical significance of differences between performance from different training stages, donwload the approximate randomization testing script from: http://www.clips.ua.ac.be/scripts/art

Then run it on the outputs from each training stage -- these are saved to pred_syntactic\results\predictions

## Harware, OS, Runtime 

The experiments were run on Ubuntu 14.02, using the following CPU: 
> 8x Intel Core i7-4810MQ CPU @ 2.80 GHz

Running train_bimodal_auto_encoder.py took approximately 20 minutes, train_softmax_model.py approximately 230 minutes.




