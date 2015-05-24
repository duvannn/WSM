from itertools import izip, repeat
from multiprocessing import Pool
from numpy import savez, load
from scipy import io, spatial, sparse
import re
import math
"""
Tools for saving, loading and testing co-occurrence matrices
"""


def save_dict(dic,      outfile):
    """
    Saves dict in npz format.
    dic: Python dictionary
    outfile: Name of the saved file
    """
    savez(outfile + '.dict.npz', dict=dic)


def load_dict(infile):
    """
    load_dict(infile) -> sparse.csr_matrix
    Loads dict from npz format.
    infile: Filename (including .npz)
    """
    return load(infile)['dict'].tolist()


def save_freqs(wordcounter, contextcounter, outfile):
    savez(outfile + '.freqs.npz', word=wordcounter, context=contextcounter)


def load_freqs(infile):
    f = load(infile)
    return f['word'].tolist(), f['context'].tolist()


def save_words(words, outfile):
    """
    Saves word counter in npz format.
    words: Counter of words.
    outfile: Name of the saved file.
    """
    savez(outfile + '.words.npz', words=words)


def load_words(infile):
    """
    load_words(infile) -> counter
    Loads word counter from npz format.
    infile: Filename (including .npz).
    """
    return load(infile)['words'].tolist()


def save_matrix_mat(matrix, outfile):
    """
    Saves matrix in mat format.
    matrix: sparse.csr_matrix or sparse.csc_matrix.
    outfile: Filename.
    """
    io.savemat(outfile + '.mat', mdict={'matrix': matrix})


def load_matrix_mat(infile):
    """
    load_matrix_mat(infile) -> matrix
    Loads matrix from mat file.
    infile: Filename (including .mat)
    """
    return io.loadmat(infile)['matrix']


def save_matrix_coo_mat(matrix, outfile):
    """
    Saves matrix in coo format (for matlab)
    matrix: numpy.array or any scipy.sparse matrix format.
    outfile: Filename.
    """
    if not sparse.isspmatrix(matrix):
        matrix = sparse.coo_matrix(matrix)
    if not sparse.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()
    m, n = matrix.shape
    io.savemat(outfile + ".mat", mdict={'i': matrix.row + 1,
                                        'j': matrix.col + 1, 's': matrix.data, 'm': m, 'n': n})


def load_matrix_coo_mat(infile):
    """
    load_matrix(infile) -> scipy.sparse.coo_matrix
    Loads coo matrix from .mat format.
    infile: Filename (indlucing .mat).
    """
    f = io.loadmat(infile)
    i = f['i'][0] - 1
    j = f['j'][0] - 1
    d = f['s'][0]
    m = f['m'][0]
    n = f['n'][0]
    return sparse.coo_matrix((d, (i, j)), shape=(m, n))


def split_file(infile, outfile, np, n):
    """
    split_file(infile,outfile,np,n)
    Splits a large text file into equally sized files.
    infile: Filename of text file.
    outfile: Filename of the saved files.
    np: Number of processes (should equal number of CPUs)
    n: Number of files.
    """
    lines = []
    with open(infile) as r:
        for line in r:
            lines.append(line)
    lines = [lines[i::n] for i in xrange(n)]
    pool = Pool(processes=np)
    pool.imap(split_file_pool, izip(lines, repeat(outfile), range(n)))
    pool.close()
    pool.join()


def split_file_pool((lines, infile, n)):
    """
    Pool function for split_file
    """
    outfile = infile + "." + str(n)
    with open(outfile, "w") as w:
        for line in lines:
            w.write(line)


def preprocess(infile, outfile):
    """
    Preprocessing for ukwac corpus, removes everything except a-Z , 0-9 and '.
    infile: Filename of unprocessed corpus.
    outfile: Filename of preprocessed corpus.
    """
    p1 = re.compile(
        r"(?:(CURRENT.*\s?))|([^a-z\s'0-9A-Z])")  # Removes CURRENT URL: line and everything except a-Z, 0-9 and '.
    # Removes spaces between ' and removes ' that are unattached to letters.
    p2 = re.compile(r"([a-z]+)\ ([a-z]?'[a-z]+)")
    p3 = re.compile(r"\ {2,}|\ +'\ +")  # Removes multiple spaces

    with open(infile) as r, open(outfile, "w") as w:
        for line in r:
            w.write(
                p3.sub(r" ", p2.sub(r"\g<1>\g<2>", p1.sub(r"", line).lower())))


def toefl_test(testfile, matrix, dic):
    """
    Toefl test.
    testfile: Filename of toefl teest.
    matrix: Co-ocurrence matrix.
    dic: Word index map for the matrix.

    """
    correctlist = []
    unknown_target = []
    unknown_answer = []
    unknown_cosine = []
    incorrect = []
    with open(testfile) as toefl:
        for line in toefl:
            flag = False
            target, correct, alt2, alt3, alt4 = line.replace(
                "(", "").replace(")", "").split()
            if target in dic:
                targetvec = matrix[dic[target], :].todense()
                if correct in dic:
                    correctvec = matrix[dic[correct], :].todense()
                    correctlen = 1 - \
                        spatial.distance.cosine(targetvec, correctvec)
                    if math.isnan(correctlen) and correctlen < 0.0:
                        unknown_cosine.append((target, correct))
                        flag = False
                    else:
                        flag = True
                        for alt in (alt2, alt3, alt4):
                            if alt in dic:
                                altvec = matrix[dic[alt], :].todense()
                                altlen = 1 - \
                                    spatial.distance.cosine(targetvec, altvec)
                                if math.isnan(correctlen):
                                    unknown_cosine.append((target, alt))
                                    flag = False
                                elif altlen > correctlen:
                                    flag = False
                            else:
                                unknown_answer.append(alt)
                else:
                    unknown_answer.append(correct)
                if flag:
                    correctlist.append((target, correct))
                else:
                    incorrect.append((target, correct))
            else:
                unknown_target.append(target)
    print "TOEFL synonym score: " + str(float(len(correctlist)) / float((len(correctlist) ≈+ len(incorrect))))
    print "Correct: " + str(correctlist)
    print "Incorrect: " + str(incorrect)
    print "Unknown cosine: " + str(unknown_cosine)
    print "Unknown targets: " + str(unknown_target)
    print "Unknown answers: " + str(unknown_answer)


def nns(word, matrix, wrddic):
    """
    Lists the nearest neighbours of a word.
    word: The word in question.
    matrix: The co-occurence matrix.
    wrddic: The word to index mapping of the matrix.
    """
    sp = False
    if sparse.isspmatrix(matrix):
        sp = True
        if not sparse.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
    res = {}

    if word in wrddic:
        if sp:
            w_vec = matrix.getrow(wrddic[word]).todense()
        else:
            w_vec = matrix[wrddic[word], :]
        for k in wrddic:
            if sp:
                k_vec = matrix.getrow(wrddic[k]).todense()
            else:
                k_vec = matrix[wrddic[k], :]
            sim = 1 - spatial.distance.cosine(w_vec, k_vec)
            if (not math.isnan(sim)) and (not math.isinf(sim)):
                res[k] = sim
    return sorted(res.iteritems(), key=lambda(k, v): v, reverse=True)[1:]


def nns_print(sorted_res, nr):
    """
    Help function to print a number of nearest neighbours.
    sorted_res: The result of the nns function.
    nr: Number of neighbours.
    """
    for r in sorted_res[:nr]:
        print str(r[0]) + ' ' + str(r[1])
