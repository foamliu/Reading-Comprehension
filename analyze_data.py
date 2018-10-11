import jieba
import matplotlib.pyplot as plt

from config import *


def analyze():
    filename = os.path.join(train_folder, train_filename)
    with open(filename, 'r') as file:
        lines = file.readlines()

    passage_lengths = []
    query_lengths = []

    for line in lines:
        item = json.loads(line)
        passage = item['passage']
        passage_lengths.append(len(list(jieba.cut(passage.strip()))))
        query = item['query']
        query_lengths.append(len(list(jieba.cut(query.strip()))))

    num_bins = 100
    n, bins, patches = plt.hist(passage_lengths, num_bins, facecolor='blue', alpha=0.5)
    plt.title('Passage Lengths Distribution')
    plt.show()

    num_bins = 100
    n, bins, patches = plt.hist(query_lengths, num_bins, facecolor='blue', alpha=0.5)
    plt.title('Query Lengths Distribution')
    plt.show()


if __name__ == '__main__':
    analyze()
