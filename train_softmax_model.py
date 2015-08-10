from Models.SoftmaxPredictor import SoftmaxPredictor
from TrainingData.data_sets.save_load_data import load_bimodal_data_set
from Eval.functions import results_to_disk, write_preds_to_file, plot_2D_embeddings, plot_metric
from Eval.functions import get_f1_and_classification_report


def run():

    for training_condition in ['context', 'context_stress', 'context_phonology']:

        # load data set with dimensionality-reduced word embeddings obtained from bimodal auto encoder
        # trained in 'train_bimodal_auto_encoder.py'
        data_set = load_bimodal_data_set(training_condition)

        # softmax model for predicting words from the right context
        softmax = SoftmaxPredictor(data_set=data_set,
                                   n_next_words=2000,
                                   learning_rate=0.008,
                                   input_size=500)
        # train the model
        softmax.train()

        # get word embeddings
        embeddings = softmax.embeddings_over_epochs[-1]

        # write predicted categories to file -- files can be used for additional processing, e.g. significance testing
        # on predictions made at two different stages in the training process
        write_preds_to_file(embeddings, '10-NN', '%s_10-NN_after_stage2' % training_condition)

        # append 10-NN results obtained from word embeddings to file
        micro_f1, macro_f1, classification_report = get_f1_and_classification_report(embeddings, '10-NN')
        results_to_disk(micro_f1, macro_f1, classification_report, epoch=softmax.epochs,
                        condition=training_condition, training_stage='AFTER STAGE 2', newfile=False)

        # plot word embeddings, reduced to two dimensions via the T-SNE algorithm
        plot_2D_embeddings(embeddings, training_condition, training_stage='after_stage_2')

        # plot accuracy over all training epochs
        plot_metric(plot_name='%s_softmax' % training_condition, plot_type='micro_f1_over_epochs',
                    ys=softmax.f1_over_epochs, label='micro_f1')

        # plot training error over all training epochs
        plot_metric(plot_name='%s_softmax' % training_condition, plot_type='training_error_over_epochs',
                    ys=softmax.training_error_over_epochs, label='training_error')


if __name__ == '__main__':
    run()