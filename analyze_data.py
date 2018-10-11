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
        seg_list = jieba.cut(passage.strip())
        length = len(list(seg_list))
        if length <= 500:
            passage_lengths.append(length)
        query = item['query']
        seg_list = jieba.cut(query.strip())
        length = len(list(seg_list))
        query_lengths.append(length)

    num_bins = 10
    n, bins, patches = plt.hist(passage_lengths, num_bins, facecolor='blue', alpha=0.5)
    plt.title('Passage Lengths Distribution')
    plt.show()

    num_bins = 10
    n, bins, patches = plt.hist(query_lengths, num_bins, facecolor='blue', alpha=0.5)
    plt.title('Query Lengths Distribution')
    plt.show()


if __name__ == '__main__':
    analyze()
