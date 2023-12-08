import numpy as np
from evaluation.evaluation_drop_f1 import get_drop_f1
from evaluation.evaluation_F1 import get_f1 as squad2_compute_f1
from evaluation.evaluation_ROUGE_L import get_rouge_l
from evaluation.evaluation_EM import get_exact_match


def evaluate(dataset, predictions, answers):
    # predictions list[str]; answers list[str]
    assert len(predictions)==len(answers), (len(predictions), len(answers))
    
    if dataset in ['squad2', 'boolq', 'boolq_np']:
        # 用squad2_compute_f1
        # return {'exact_match': exact_match, 'f1': f1}
        eval = squad2_compute_f1(predictions, answers)
    elif dataset == 'drop':
        # 用drop_f1_metrics
        # return {'exact_match': exact_match, 'f1': f1}
        eval = get_drop_f1(predictions, answers)
    elif dataset in ['newsqa', 'narrativeqa']:
        eval = get_rouge_l(predictions, answers)
    else:
        ems = []
        for (prediction, ans) in zip(predictions, answers):
            ems.append(get_exact_match(prediction, ans))
        eval = {'exact_match': np.mean(ems)}
    return eval