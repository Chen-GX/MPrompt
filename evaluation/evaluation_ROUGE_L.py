import copy
import json
import rouge
import logging
# from rouge_score import rouge
logger = logging.getLogger(__name__)

rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
    # max_n=4,
    # limit_length=True,
    # length_limit=100,
    # length_limit_type="words",
    # apply_avg=True,
    # apply_best=True,
    # alpha=0.5,
    # weight_factor=1.2,
    # stemming=True,
)

def rouge_l(p, g):
    try:
        return rouge_l_evaluator.get_scores(p, g, avg=True)
    except:
        logger.info(f'error rouge_l predict: {p}')
        return {'rouge-l':{'r': 0.0, 'p': 0.0, 'f': 0.0}}

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction.lower(), ground_truth.lower())
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

    
def get_rouge_l(predictions, golds):
    score = total = 0
    for prediction, gold in zip(predictions, golds):
        rouge_l_score = rouge_l(prediction.lower(), gold.lower())
        score += rouge_l_score["rouge-l"]["f"]
        total += 1
    score = score / total
    return {'rouge_l': score}
