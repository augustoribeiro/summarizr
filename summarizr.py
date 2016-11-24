import math
import nltk.data
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

sentences_detector = nltk.data.load('tokenizers/punkt/english.pickle')

text = '''
Breast cancer is an increasing public health problem.
Substantial advances have been made in the treatment of breast cancer,
but the introduction of methods to predict women at elevated risk and prevent the disease has been less successful.
Here, we summarize recent data on newer approaches to risk prediction, available approaches to prevention,
how new approaches may be made, and the difficult problem of using what we already know to prevent breast cancer in
populations. During 2012, the Breast Cancer Campaign facilitated a series of workshops, each covering a specialty
area of breast cancer to identify gaps in our knowledge. The risk-and-prevention panel involved in this exercise was
asked to expand and update its report and review recent relevant peer-reviewed literature. The enlarged position paper
presented here highlights the key gaps in risk-and-prevention research that were identified, together with
recommendations for action. The panel estimated from the relevant literature that potentially 50% of breast cancer
could be prevented in the subgroup of women at high and moderate risk of breast cancer by using current chemoprevention
(tamoxifen, raloxifene, exemestane, and anastrozole) and that, in all women, lifestyle measures,
including weight control, exercise, and moderating alcohol intake, could reduce breast cancer risk by about 30%.
Risk may be estimated by standard models potentially with the addition of, for example, mammographic density and
appropriate single-nucleotide polymorphisms. This review expands on four areas:
(a) the prediction of breast cancer risk, (b) the evidence for the effectiveness of preventive therapy and lifestyle
approaches to prevention, (c) how understanding the biology of the breast may lead to new targets for prevention,
and (d) a summary of published guidelines for preventive approaches and measures required for their implementation.
We hope that efforts to fill these and other gaps will lead to considerable advances in our efforts to predict
risk and prevent breast cancer over the next 10 years. '''


def frequencies(tokenized_sentences):
    wordcount = defaultdict(int)
    for sentence in tokenized_sentences:
        for word in sentence:
            wordcount[word] += 1
    return wordcount


def calculate_tf(word_frequencies):
    word_tfs = {}
    for word in word_frequencies:
        word_tfs[word] = 1 + math.log(word_frequencies[word])
    return word_tfs


def calculate_score(sentence, word_tfs):
    score = 0
    for word in remove_stopwords(sentence):
        score += word_tfs[word]
    return score


def remove_stopwords(tokens):
    return [token for token in tokens if
            token not in stopwords.words('english') and token not in string.punctuation]


def summarize(text, sentence_number=5):
    sentences = sentences_detector.tokenize(text.strip())
    tokenized_sentences = [remove_stopwords(word_tokenize(sentence)) for sentence in sentences]

    word_frequencies = frequencies(tokenized_sentences)
    word_tfs = calculate_tf(word_frequencies)

    scored_sentences = [(ind, calculate_score(p, word_tfs)) for ind, p in enumerate(tokenized_sentences)]
    sorted_scored_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    selected_sentences = sorted(sorted_scored_sentences[0:sentence_number], key=lambda x: x[0])

    summary = [sentences[i] for i in [ind for ind, score in selected_sentences]]
    print '\n'.join(summary)


summarize(text, sentence_number=3)
