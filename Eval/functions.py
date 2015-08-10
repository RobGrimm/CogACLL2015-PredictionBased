import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score
from matplotlib import pyplot

# set parameters for plots
pyplot.rcParams.update({'figure.figsize': (25, 20), 'font.size': 10})
# define directory for storing results
save_results_to_dir = os.path.abspath(os.path.dirname(__file__)).rstrip('/Eval') + '/results/'


########################################################################################################################

# helper functions


def get_pos_tag(word):
    # a word is a string 'word-'pos_tag'
    # this returns the pos tag
    return word.split('-')[1]


def get_pos_tags(words):
    return [get_pos_tag(w) for w in words]


def get_paras_for_centering_legend_below_plot():
    # get matplotlib parameters for centering the legend below plots
    pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    lgd = pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    art = [lgd]
    return art


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_plot_to(plot_dir, plot_name, create_folder_if_not_exists=True):
    if create_folder_if_not_exists:
        create_dir_if_not_exists(plot_dir)

    pyplot.savefig(plot_dir + plot_name, additional_artists=get_paras_for_centering_legend_below_plot(),
                   bbox_inches='tight')
    pyplot.close()


def create_graph(x, y, marker, label, e=None):
    # create custom matplotlib plot
    assert len(x) == len(y)
    if e is None:
        pyplot.plot(x, y, marker, markersize=40, linewidth=9, label=label)
    else:
        pyplot.errorbar(x, y, e,  markersize=40, linewidth=9, label=label)
    pyplot.rcParams.update({'font.size': 50})


def plot_metric(plot_name, plot_type, ys, label, error=None):
    xs = range(len(ys))
    create_graph(xs, ys, marker='go-', label=label, e=error)
    plot_dir = save_results_to_dir + '/%s/' % plot_type
    save_plot_to(plot_dir, plot_name)


########################################################################################################################

# functions for: retrieving results from trained models, plotting results, saving results to disk


def get_f1_and_classification_report(embeddings_dict, classifier):
    xs, ys, y_pred = get_xs_ys_predictions(embeddings_dict, classifier)
    class_names = ['verbs', 'nouns', 'adjectives', 'closed class words']
    report = classification_report(y_true=ys, y_pred=y_pred, target_names=class_names)
    micro_f1 = f1_score(y_true=ys, y_pred=y_pred, average='micro')
    macro_f1 = f1_score(y_true=ys, y_pred=y_pred, average='macro')
    return micro_f1, macro_f1, report


def get_xs_ys_predictions(embeddings_dict, classifier):
    """
    Run a classifier of type 'classifier' (one of: majority vote baseline,
    tratified sampling baseline, 10-NN classifier).

    Return:
        - xs: the word embeddings
        - ys: the gold standard labels
        - y_pred: the predicted labels
    """
    assert classifier in ['majority_vote', 'stratified', '10-NN']

    pos_ints = {'v': 0, 'n': 1, 'adj': 2, 'fn': 3}

    ys = []
    xs = []

    words = sorted(embeddings_dict.keys())
    for w in words:
        xs.append(embeddings_dict[w])
        # get embeddings's pos tag, look up pos tag's unique integer
        label = pos_ints[get_pos_tag(w)]
        ys.append(label)

    clf = None
    if classifier == 'majority_vote':
        clf = DummyClassifier(strategy='most_frequent', random_state=0)
    elif classifier == 'stratified':
        clf = DummyClassifier(strategy='stratified', random_state=0)
    elif classifier == '10-NN':
        clf = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree')

    clf.fit(xs, ys)
    y_pred = clf.predict(xs)

    return xs, ys, y_pred


def write_preds_to_file(embeddings_dict, classfier, outfile_name):
    """
    Write predictions made by 'classifier' and gold standard labels to file.
    Files can be used for further processing -- e.g. to compare predictions made by different classifiers.
    """
    results_dir = save_results_to_dir + '/predictions/'
    create_dir_if_not_exists(results_dir)
    xs, ys, ys_pred = get_xs_ys_predictions(embeddings_dict, classfier)

    with open('%s%s' % (results_dir, outfile_name), 'w') as outfile:
        for x, y, y_pred in zip(range(len(xs)), ys, ys_pred):
            outfile.write('%s %s %s\n' % (x, y, y_pred))


def plot_2D_embeddings(embeddings_dict, condition, training_stage):
    """
    Take word embeddings from last epoch. Reduce them to 2 dimensions using the TSNE algorithm.
    Create two plots and save to disk:

        - colored embeddings: color each data point by syntactic type
        - orthographic embeddings: plot each data point as the word's orthographic word form
     """
    # set readable font size for orthographic embeddings
    pyplot.rcParams.update({'font.size': 10})

    tsne = TSNE(n_components=2)
    color_maps = {'v': pyplot.get_cmap("Blues"), 'n': pyplot.get_cmap("Reds"), 'adj': pyplot.get_cmap("Greens"),
                  'fn': pyplot.get_cmap('Greys')}

    words = embeddings_dict.keys()
    vectors = embeddings_dict.values()
    pos_tags = get_pos_tags(words)
    reduced_data = tsne.fit_transform(np.array(vectors))

    # plot embeddings as data points that are colored by syntactic class
    for xy, pos in zip(reduced_data, pos_tags):
        pyplot.plot(xy[0], xy[1], 'o', markersize=20, color=color_maps[pos](0.7))

    # the directory for the plots
    plot_dir = save_results_to_dir + '/t_sne_color_embeddings/'
    # the name of the plot file
    plot_name = '%s_%s.png' % (condition, training_stage)
    save_plot_to(plot_dir, plot_name)

    # plot plain words
    fig = pyplot.figure()
    ax = fig.add_subplot(111)

     # plot embeddings as orthographic word forms
    for i, j in zip(reduced_data, words):
        pyplot.plot(i[0], i[1])
        ax.annotate(j, xy=i)

    plot_dir = save_results_to_dir + '/t_sne_orthographic_embeddings/'
    save_plot_to(plot_dir, plot_name)


def results_to_disk(micro_f1, macro_f1, classification_report, epoch, condition, training_stage, newfile):
    """
    Write results to file.

    Either create a new file (newfile=True) or append to an existing file (newfile=False).
    """
    results_dir = save_results_to_dir + '/results_over_training_stages/'
    create_dir_if_not_exists(results_dir)

    if newfile:
        # write to new file
        mode = 'w'
    else:
        # append to existing file
        mode = 'a'

    with open('%s%s.txt' % (results_dir, condition), mode) as outfile:
        outfile.write('%s\n\n' % training_stage)
        outfile.write('epoch: %s\n' % epoch)
        outfile.write(classification_report)
        outfile.write('\n\n')
        outfile.write('10-NN micro F1: %s\n' % micro_f1)
        outfile.write('10-NN macro F1: %s\n' % macro_f1)
        outfile.write('\n\n\n')