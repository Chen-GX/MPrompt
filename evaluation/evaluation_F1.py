# read lines and calculate F1/EM
import collections
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)  # 统计共有部分
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def get_f1(predictions, gold):
    no_ans = "no answer"
    f1 = exact_match = total = 0
    for i, prediction in enumerate(predictions):
        # For unanswerable questions, only correct answer is empty string
        if no_ans in gold[i].lower():
            gold[i] = ""

        if no_ans in prediction.lower():
            prediction = ""

        exact_match += compute_exact(gold[i], prediction)
        f1 += compute_f1(gold[i], prediction)

        total += 1

    exact_match = exact_match / total
    f1 = f1 / total
    eval = {'exact_match': exact_match, 'f1': f1}
    return eval