from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def n_gram_precision(
    reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """

    ref_ngrams = set(grouper(reference, n))
    cand_ngrams = grouper(candidate, n)

    if not cand_ngrams:
        return 0.0
    
    correct = 0
    for ng in cand_ngrams:
        if ng in ref_ngrams:
            correct += 1

    return correct / len(cand_ngrams)


def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """

    ref_len = len(reference)
    cand_len = len(candidate)

    if cand_len == 0:
        return 0.0
    
    if cand_len > ref_len:
        return 1
    else:
        return exp(1 - ref_len / cand_len)


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    """
    Calculate the BLEU score.  Please scale the BLEU score by 100.0

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """

    p_ns = [n_gram_precision(reference, candidate, i) for i in range(1, n+1)]

    if any(p == 0 for p in p_ns):
        return 0.0
    
    result = 1
    for p in p_ns:
        if p > 0:
            result = result * p

    bp = brevity_penalty(reference, candidate)
    bleu = bp * result ** (1/n)
    return bleu * 100.0
