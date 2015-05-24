from collections import Counter
from scipy import sparse
import numpy
from itertools import izip, repeat
from multiprocessing import Pool
from utils import save_dict, save_words, save_matrix_mat, load_matrix_mat, split_file, toefl_test, save_matrix_coo_mat, save_freqs


def count_freqs(infile, word_count):
    """
    Counts the frequency of words in a file.
    infile: Filename of text.
    word_count: Number of words to return.
    """
    words = Counter()
    inp = open(infile, "r")
    for line in inp.readlines():
        for wrd in line.split():
            words[wrd] += 1
    inp.close()
    return words.most_common(word_count)


def word_vector(words):
    wrddict = {}
    cnt = 0
    for word in words:
        wrddict[word[0]] = cnt
        cnt += 1
    return wrddict


def word_list(infile):
    lines = []
    with open(infile) as textin:
        for line in textin:
            lines.append(line)
    return lines


def wsm(line, wrddict, win, matrix, ctxcounter, wrdcounter):
    """
    Use the build function.
    """
    if not (isinstance(line, str)):
        raise TypeError("NOT A STRING")
    words = line.split()
    cnt = 0
    for word in words:
        if word in wrddict:
            wrdcounter[word] += 1
            ctx = 1
            while ctx <= win:
                if (cnt + ctx) < len(words):
                    ctxword = words[cnt + ctx]
                    if ctxword in wrddict:
                        matrix[wrddict[word], wrddict[ctxword]] += 1
                        ctxcounter[ctxword] += 1
                        matrix[wrddict[ctxword], wrddict[word]] += 1
                        ctxcounter[word] += 1
                    ctx += 1
                else:
                    break
        cnt += 1
    del words, line, cnt, wrddict
    return matrix, wrdcounter, ctxcounter


def wsm_builder((lines, wrddict, win)):
    """
    Use the build function.
    """
    matrix = sparse.lil_matrix((len(wrddict), len(wrddict)))
    ctxcounter = Counter()
    wrdcounter = Counter()
    if (isinstance(lines, str)):
        matrix, wrdcounter, ctxcounter = wsm(
            lines, wrddict, win, matrix, ctxcounter, wrdcounter)
    else:
        for line in lines:
            matrix, wrdcounter, ctxcounter = wsm(
                line, wrddict, win, matrix, ctxcounter, wrdcounter)
    del lines, wrddict, line, win
    return matrix.tocsr(), wrdcounter, ctxcounter


def freqs_count((lines, wrddict, win)):
    """
    Counts words and context words of an list of strings.
    lines: List of sentences.
    wrddict: Word index mapping.
    win: Word context window size.
    """
    ctxcounter = Counter()
    wrdcounter = Counter()
    if (isinstance(lines, str)):
        wrdcounter, ctxcounter = freq_count(
            lines, wrddict, win, ctxcounter, wrdcounter)
    else:
        for line in lines:
            wrdcounter, ctxcounter = freq_count(
                line, wrddict, win, ctxcounter, wrdcounter)
    return wrdcounter, ctxcounter


def freq_count(line, wrddict, win, ctxcounter, wrdcounter):
    """
    Counts words and context words of a string.
    line: The sentence as a string.
    wrddict: Word index mapping.
    win: Word context window size.
    ctxcounter: Context Counter.
    wrdcounter: Word Counter.
    """
    if not (isinstance(line, str)):
        raise TypeError("NOT A STRING")
    words = line.split()
    cnt = 0
    for word in words:
        if word in wrddict:
            wrdcounter[word] += 1
            ctx = 1
            while ctx <= win:
                if (cnt + ctx) < len(words):
                    ctxword = words[cnt + ctx]
                    if ctxword in wrddict:
                        ctxcounter[ctxword] += 1
                        ctxcounter[word] += 1
                    ctx += 1
                else:
                    break
        cnt += 1
    return wrdcounter, ctxcounter


def toefl_words(testfile, wrddict):
    """
    Makes sure every word in the toefl test is in the dictionary.
    testfile: Toefl test filename.
    wrddict: word to index mapping.
    """
    toefls = []
    with open(testfile) as toefl:
        for line in toefl:
            toefls.extend(line.replace("(", "").replace(")", "").split())
    cnt = len(wrddict)
    for word in toefls:
        if word not in wrddict:
            wrddict[word] = cnt
            cnt += 1
    return wrddict


def ppmi(matrix):
    """
    Performs PPMI on the co-occurence matrix.
    matrix: co-occurence matrix.
    """
    if not sparse.isspmatrix(matrix):
        sparse.coo_matrix(matrix)
    elif not sparse.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()
    row = matrix.row
    col = matrix.col
    msum = matrix.sum()
    csum = matrix.sum(0).A1
    rsum = matrix.sum(1).A1
    s = numpy.array([rsum[r] * csum[c] for r, c in izip(row, col)])
    matrix.data = numpy.log2(matrix.data * msum / s)
    matrix.data[matrix.data < 0.0] = 0
    matrix.data[numpy.isnan(matrix.data)] = 0
    return matrix


def build(infile, outfile, testfile, word_count, word_window, np):
    """
    infile: Filename of text file.
    outfile: Name of the saved files.
    testfile: Filename of toefl test.
    word_count: Number of words used to build the matrix.
    word_window: Size of context window.
    np: Number of pools (Processes) to be used.
    """
    words = count_freqs(infile, word_count)
    wrddic = word_vector(words)
    wrddic = toefl_words(testfile, wrddic)
    lines = word_list(infile)

    ctxcounter = Counter()
    wrdcounter = Counter()
    matrix = sparse.csr_matrix((len(wrddic), len(wrddic)))

    if np == 1:
        matrix, wrdcounter, ctxcounter = wsm_builder(
            (lines, wrddic, word_window))
    else:
        if np > len(lines):
            np = len(lines)
        lines = [lines[i::np] for i in xrange(np)]
        pool = Pool(processes=np)
        pools = pool.imap(
            wsm_builder, izip(lines, repeat(wrddic.copy()), repeat(word_window)))
        pool.close()
        pool.join()

        for m, w, c in pools:
            matrix = matrix + m
            wrdcounter = wrdcounter + w
            ctxcounter = ctxcounter + c

    save_words(ctxcounter, outfile + "CTX")
    save_words(wrdcounter, outfile + "WRD")
    save_words(words, outfile)
    save_dict(wrddic, outfile)
    save_matrix_mat(matrix, outfile)
