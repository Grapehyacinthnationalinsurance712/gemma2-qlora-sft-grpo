"""
Evaluation utilities: per-sample scoring and aggregate comparison.
"""

from typing import Dict, List, Optional

from src.rewards.correctness import correctness_reward, extract_gsm8k_final_answer
from src.rewards.format import format_reward


def score_response(
    response: str,
    ground_truth: str,
    use_style: bool = False,
    classifier_model=None,
    classifier_tokenizer=None,
) -> Dict[str, float]:
    """Score a single response on all reward components.

    Args:
        response: Model-generated response string.
        ground_truth: Ground-truth answer string.
        use_style: Whether to compute the style reward.
        classifier_model: Yoda style classifier model.
        classifier_tokenizer: Corresponding tokenizer.

    Returns:
        Dict with keys: Correctness, Format, Style (optional), Total.
    """
    scores = {
        "Correctness": correctness_reward([""], [response], answer=[ground_truth])[0],
        "Format":      format_reward([""], [response])[0],
    }

    if use_style and classifier_model is not None:
        from src.rewards.style import style_reward
        scores["Style"] = style_reward(
            [""], [response],
            classifier_model=classifier_model,
            classifier_tokenizer=classifier_tokenizer,
        )[0]

    scores["Total"] = sum(scores.values())
    return scores


def compare_before_after(
    questions: List[str],
    ground_truths: List[str],
    before_responses: List[str],
    after_responses: List[str],
    before_label: str = "Before",
    after_label: str = "After",
    use_style: bool = False,
    classifier_model=None,
    classifier_tokenizer=None,
) -> Dict[str, Dict[str, float]]:
    """Compute and print average reward scores before and after training.

    Args:
        questions: List of input questions.
        ground_truths: Corresponding ground-truth answers.
        before_responses: Model outputs before training.
        after_responses: Model outputs after training.
        before_label: Display label for the before model.
        after_label: Display label for the after model.
        use_style: Whether to include style reward.
        classifier_model: Yoda style classifier.
        classifier_tokenizer: Corresponding tokenizer.

    Returns:
        Dict with 'before' and 'after' average score dicts.
    """
    scores_before, scores_after = [], []
    reward_cols = ["Correctness", "Format"] + (["Style"] if use_style else []) + ["Total"]

    for q, gt, bef, aft in zip(questions, ground_truths, before_responses, after_responses):
        sb = score_response(bef, gt, use_style, classifier_model, classifier_tokenizer)
        sa = score_response(aft, gt, use_style, classifier_model, classifier_tokenizer)
        scores_before.append(sb)
        scores_after.append(sa)

        gt_num = extract_gsm8k_final_answer(gt)
        print(f"\n{'─' * 70}")
        print(f"Q: {q[:100]}...")
        print(f"Ground truth answer: {gt_num}")
        print(f"\n  [{before_label}]: {bef[:200]}...")
        print(f"  [{after_label}]:  {aft[:200]}...")
        header = "  " + " " * 20 + " | " + " ".join(f"{c:>12}" for c in reward_cols)
        print(header)
        print("  " + "─" * len(header))
        print(f"  {before_label:>20} | " + " ".join(f"{sb[c]:>12.3f}" for c in reward_cols))
        print(f"  {after_label:>20} | " + " ".join(f"{sa[c]:>12.3f}" for c in reward_cols))

    n = len(scores_before)
    avg_b = {c: sum(s[c] for s in scores_before) / n for c in reward_cols}
    avg_a = {c: sum(s[c] for s in scores_after)  / n for c in reward_cols}

    print(f"\n{'=' * 70}")
    print(f"AVERAGE over {n} examples:")
    header = " " * 22 + " | " + " ".join(f"{c:>12}" for c in reward_cols)
    print(header)
    print("─" * len(header))
    print(f"{before_label:>22} | " + " ".join(f"{avg_b[c]:>12.3f}" for c in reward_cols))
    print(f"{after_label:>22} | "  + " ".join(f"{avg_a[c]:>12.3f}" for c in reward_cols))
    delta = {c: avg_a[c] - avg_b[c] for c in reward_cols}
    print(f"{'Delta':>22} | " + " ".join(f"{delta[c]:>+12.3f}" for c in reward_cols))

    return {"before": avg_b, "after": avg_a}