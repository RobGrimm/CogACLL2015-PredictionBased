from Models.PhonAutoEncoder import PhonAutoEncoder
from Models.BiModalAutoEncoder import BiModalAutoEncoder
from Eval.functions import plot_2D_embeddings, plot_metric, write_preds_to_file, results_to_disk
from Eval.functions import get_f1_and_classification_report
from TrainingData.data_sets.save_load_data import save_bimodal_data_set, load_text_data_set
from TrainingData.load_corpus.load_corpus_data import get_target_words
from Phonology.get_phonology_vectors import get_primary_stress_vectors


def run():

    data_set = load_text_data_set('CDS')

    # exclude the following words, all longer than three syllables, which is not supported by the system ued to
    # construct phonological feature vectors
    more_than_three_syllables = ['actually', 'anybody', 'investigator', 'helicopter', 'interesting', 'cookie_monster',
                                 'definitely', 'refrigerator',  'oh_my_goodness',  'humpty_dumpty', 'interested',
                                 'everybody', 'father_christmas', 'helicopter', 'alligator', 'caterpillar', "everybody's",
                                 'hippopotamus']

    # also exclude the empty string, which occasionally occurs in the corpus
    empty_string = ['']

    vocabulary = get_target_words(2000, tags={'n', 'v', 'adj', 'fn'},
                                   exclude_words=more_than_three_syllables + empty_string)

    # auto encoder for reducing the dimensionality of the phonological feature vectors
    phon_ae = PhonAutoEncoder(vocabulary=vocabulary,
                              epochs=200,
                              learning_rate=0.1,
                              n_hidden=30,
                              corruption_level=0.1,
                              batch_size=1)

    phon_ae.train()

    for training_condition in ['context', 'context_stress', 'context_phonology']:

        # if 'phonology' is part of the model name, get dimensionality-reduced phonological feature vectors
        if 'phonology' in training_condition:
            phon_vs = phon_ae.get_hidden_vectors()
            phon_vs_size = 30
        else:
            phon_vs = None
            phon_vs_size = 0

        # if 'stress' is part of the model name, get dimensionality-reduced lexical stress feature vectors
        if 'stress' in training_condition:
            stress_vs = get_primary_stress_vectors(vocabulary)
            stress_vs_size = 3
        else:
            stress_vs = None
            stress_vs_size = 0

        # auto encoder for projecting left-context and phonological / lexical stress feature vector into a shared
        # dimensionality-reduced space
        bm_ae = BiModalAutoEncoder(data_set=data_set,
                                   context_feature_size=2000,
                                   phon_vectors=phon_vs,
                                   phon_feature_size=phon_vs_size,
                                   stress_vectors=stress_vs,
                                   stress_feature_size=stress_vs_size,
                                   learning_rate=0.01,
                                   n_hidden=500,
                                   corruption_level=0.1,
                                   batch_size=1)

        # vectors of left-context frequencies, plus phonological and / or lexical stress features
        input_embeddings = dict(zip(bm_ae.vocabulary, bm_ae.embeddings_matrix.get_value()))

        # write predicted categories to file -- files can be used for additional processing, e.g. significance testing
        # on predictions made at two different stages in the training process
        write_preds_to_file(input_embeddings, 'majority_vote', '%s_majority_vote' % training_condition)
        write_preds_to_file(input_embeddings, 'stratified', '%s_stratified_sampling' % training_condition)
        write_preds_to_file(input_embeddings, '10-NN', '%s_10-NN_before_stage1' % training_condition)

        # create a new file with results by training stage -- write majority vote baseline results to this file
        micro_f1, macro_f1, classification_report = get_f1_and_classification_report(input_embeddings, 'majority_vote')
        results_to_disk(micro_f1, macro_f1, classification_report, epoch='model was not trained at this stage',
                        condition=training_condition, training_stage='BASELINE 1: MAJORITY VOTE', newfile=True)

        # append stratified sampling baseline results to file
        micro_f1, macro_f1, classification_report = get_f1_and_classification_report(input_embeddings, 'stratified')
        results_to_disk(micro_f1, macro_f1, classification_report, epoch='model was not trained at this stage',
                        condition=training_condition, training_stage='BASELINE 2: STRATIFIED SAMPLING', newfile=False)

        # append 10-NN results obtained from input vectors to file
        micro_f1, macro_f1, classification_report = get_f1_and_classification_report(input_embeddings, '10-NN')
        results_to_disk(micro_f1, macro_f1, classification_report, epoch='model was not trained at this stage',
                        condition=training_condition, training_stage='BEFORE STAGE 1', newfile=False)

        # train the bimodal auto encoder
        bm_ae.train()

        # get the dimensionality-reduced embeddings after training (vectors of hidden unit activation values)
        embeddings = bm_ae.embeddings_over_epochs[-1]

        # # append 10-NN results obtained from hidden embeddings to file
        micro_f1, macro_f1, classification_report = get_f1_and_classification_report(embeddings, '10-NN')
        results_to_disk(micro_f1, macro_f1, classification_report, epoch=bm_ae.epochs,
                        condition=training_condition, training_stage='AFTER STAGE 1', newfile=False)

        # plot word embeddings, reduced to two dimensions via the T-SNE algorithm
        plot_2D_embeddings(embeddings, training_condition, training_stage='after_stage_1')

        # plot accuracy over all training epochs
        plot_metric(plot_name='%s_auto_encoder' % training_condition, plot_type='micro_f1_over_epochs',
                    ys=bm_ae.f1_over_epochs, label='micro_f1')

        # plot training error over all training epochs
        plot_metric(plot_name='%s_auto_encoder' % training_condition, plot_type='training_error_over_epochs',
                    ys=bm_ae.training_error_over_epochs, label='training_error')

        # create and save a new data set with the new embeddings -- for further training with the softmax model
        save_bimodal_data_set(bm_ae, data_set_name=training_condition, embeddings_dict=embeddings)


if __name__ == '__main__':
    run()