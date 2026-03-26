"""
Sequence Alignment Algorithms
==============================

Sequence alignment is fundamental in bioinformatics for comparing DNA, RNA,
or protein sequences. Dynamic programming algorithms find optimal alignments
by building a scoring matrix that tracks the best alignment up to each pair
of positions.

Tasks
-----
1. Implement the Needleman-Wunsch algorithm for global alignment:
   - Build the scoring matrix using match/mismatch scores and gap penalties.
   - The recurrence: F(i,j) = max(F(i-1,j-1) + s(a_i, b_j),
                                   F(i-1,j) + gap_penalty,
                                   F(i,j-1) + gap_penalty)
   - Initialize first row and column with cumulative gap penalties.

2. Implement the Smith-Waterman algorithm for local alignment:
   - Similar to Needleman-Wunsch but with F(i,j) >= 0 (add 0 to the max).
   - Traceback starts from the maximum score in the matrix.

3. Implement scoring matrices:
   - Simple match/mismatch scoring (e.g., +1 match, -1 mismatch, -2 gap).
   - BLOSUM-like scoring for amino acid sequences (simplified version).

4. Implement traceback to recover the optimal alignment string from the
   scoring matrix. Return aligned sequences with gaps indicated by '-'.

5. Compare alignment scores for several test sequence pairs. Show how
   different scoring parameters affect the alignment.
"""


def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Global sequence alignment using Needleman-Wunsch algorithm.

    Parameters
    ----------
    seq1 : str
        First sequence.
    seq2 : str
        Second sequence.
    match : int
        Score for matching characters.
    mismatch : int
        Score for mismatching characters.
    gap : int
        Gap penalty (negative value).

    Returns
    -------
    score : int
        Optimal global alignment score.
    score_matrix : list of list of int
        The full scoring matrix of shape (len(seq1)+1, len(seq2)+1).
    aligned_seq1 : str
        First sequence with gaps inserted.
    aligned_seq2 : str
        Second sequence with gaps inserted.
    """
    raise NotImplementedError


def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Local sequence alignment using Smith-Waterman algorithm.

    Parameters
    ----------
    seq1 : str
        First sequence.
    seq2 : str
        Second sequence.
    match : int
        Score for matching characters.
    mismatch : int
        Score for mismatching characters.
    gap : int
        Gap penalty (negative value).

    Returns
    -------
    score : int
        Optimal local alignment score.
    score_matrix : list of list of int
        The full scoring matrix.
    aligned_seq1 : str
        Locally aligned portion of first sequence.
    aligned_seq2 : str
        Locally aligned portion of second sequence.
    start_pos : tuple
        (i, j) position in the original sequences where local alignment starts.
    """
    raise NotImplementedError


def scoring_function(a, b, match=1, mismatch=-1):
    """
    Simple match/mismatch scoring for two characters.

    Parameters
    ----------
    a : str
        Character from first sequence.
    b : str
        Character from second sequence.
    match : int
        Score if a == b.
    mismatch : int
        Score if a != b.

    Returns
    -------
    int
        Score for aligning a with b.
    """
    raise NotImplementedError


def blosum_like_score(a, b):
    """
    Simplified BLOSUM-like scoring for amino acid pairs.

    Uses a reduced scoring matrix for common amino acids. Similar amino acids
    (e.g., I/L, D/E) receive positive scores; dissimilar pairs receive
    negative scores.

    Parameters
    ----------
    a : str
        Amino acid character.
    b : str
        Amino acid character.

    Returns
    -------
    int
        Score for aligning amino acids a and b.
    """
    raise NotImplementedError


def traceback_global(score_matrix, seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Traceback through the Needleman-Wunsch score matrix to find optimal alignment.

    Parameters
    ----------
    score_matrix : list of list of int
        Scoring matrix from Needleman-Wunsch.
    seq1 : str
        First sequence.
    seq2 : str
        Second sequence.
    match : int
        Match score.
    mismatch : int
        Mismatch score.
    gap : int
        Gap penalty.

    Returns
    -------
    aligned_seq1 : str
        Aligned first sequence.
    aligned_seq2 : str
        Aligned second sequence.
    """
    raise NotImplementedError


def traceback_local(score_matrix, seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Traceback through the Smith-Waterman score matrix for local alignment.

    Parameters
    ----------
    score_matrix : list of list of int
        Scoring matrix from Smith-Waterman.
    seq1 : str
        First sequence.
    seq2 : str
        Second sequence.
    match : int
        Match score.
    mismatch : int
        Mismatch score.
    gap : int
        Gap penalty.

    Returns
    -------
    aligned_seq1 : str
        Locally aligned portion of first sequence.
    aligned_seq2 : str
        Locally aligned portion of second sequence.
    start_pos : tuple
        Starting position in original sequences.
    """
    raise NotImplementedError


def print_alignment(aligned_seq1, aligned_seq2):
    """
    Print an alignment in a readable format with match indicators.

    Parameters
    ----------
    aligned_seq1 : str
        First aligned sequence.
    aligned_seq2 : str
        Second aligned sequence.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Task 1: Needleman-Wunsch global alignment
    seq1 = "GATTACA"
    seq2 = "GCATGCU"
    score, matrix, aln1, aln2 = needleman_wunsch(seq1, seq2)
    print("=== Needleman-Wunsch (Global Alignment) ===")
    print(f"Sequences: {seq1} vs {seq2}")
    print(f"Score: {score}")
    print_alignment(aln1, aln2)

    # Task 2: Smith-Waterman local alignment
    seq3 = "GGTTGACTA"
    seq4 = "TGTTACGG"
    score_local, matrix_local, aln3, aln4, start = smith_waterman(seq3, seq4)
    print("\n=== Smith-Waterman (Local Alignment) ===")
    print(f"Sequences: {seq3} vs {seq4}")
    print(f"Score: {score_local}")
    print(f"Local alignment starts at position: {start}")
    print_alignment(aln3, aln4)

    # Task 5: Compare scoring parameters
    print("\n=== Effect of Scoring Parameters ===")
    for gap_pen in [-1, -2, -3]:
        s, _, a1, a2 = needleman_wunsch(seq1, seq2, gap=gap_pen)
        print(f"Gap penalty = {gap_pen}: score = {s}")
        print_alignment(a1, a2)
        print()

    # Longer sequences
    seq5 = "AGTACGCA"
    seq6 = "TATGC"
    score5, _, aln5, aln6 = needleman_wunsch(seq5, seq6)
    print(f"Global alignment of {seq5} vs {seq6}: score = {score5}")
    print_alignment(aln5, aln6)
