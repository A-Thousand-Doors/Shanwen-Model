import re


def extract_solution(solution_str):
    """
    Extract the solution from the given solution string.
    Args:
        solution_str (str): The solution string to be processed.
    Returns:
        str: The extracted solution.
    """
    solution = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if solution:
        return solution.group(1).strip().replace(",", "").replace("$", "")


def compute_score_answer(solution_str, ground_truth, match_score=1.0):
    """
    Compute the score for a given solution string and ground truth.
    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The correct answer to compare against.
        match_score (float): The score to assign for a correct answer.
    Returns:
        float: The computed score.
    """
    final_solution = extract_solution(solution_str)
    return match_score if final_solution == ground_truth else 0.0
    

def compute_score_CoT_format(solution_str, match_score=1.0):
    """
    Compute the score for a given solution string based on its format.
    Args:
        solution_str (str): The solution string to be evaluated.
        match_score (float): The score to assign for a correct format.
    Returns:
        float: The computed format score.
    """
    format_match = re.search(
        r'<think>.*?</think>\s*'
        r'<answer>.*?</answer>\s*',
        solution_str,
        re.DOTALL
    )
    return match_score if format_match else 0.0


def compute_CoT_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute the score for a given solution string based on its content and format.
    Args:
        data_source (str): The source of the data.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The correct answer to compare against.
        extra_info (dict, optional): Additional information for debugging or logging.
    Returns:
        float: The computed score.
    """
    format_score = compute_score_CoT_format(solution_str, match_score=1.0)
    answer_score = compute_score_answer(solution_str, ground_truth, match_score=2.0)

    return format_score + answer_score


def compute_score_WHWM_format(solution_str, match_score=1.0):
    """
    Compute the score for a given solution string based on its format.
    Args:
        solution_str (str): The solution string to be evaluated.
        match_score (float): The score to assign for a correct format.
    Returns:
        float: The computed format score.
    """
    score = 0.0

    # 检查是否包含 <think>...</think> 后跟 <answer>...</answer>
    if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", solution_str, re.DOTALL):
        score += 0.2

    # 提取 <think>...</think> 的内容
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)

        if re.search(r"<what>.*?</what>", think_content, re.DOTALL):
            score += 0.2
        if re.search(r"<how>.*?</how>", think_content, re.DOTALL):
            score += 0.2
        if re.search(r"<why>.*?</why>", think_content, re.DOTALL):
            score += 0.2
        if re.search(r"<meaningful>.*?</meaningful>", think_content, re.DOTALL):
            score += 0.2

    return score * match_score


def compute_WHWM_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute the score for a given solution string based on its content and format.
    Args:
        data_source (str): The source of the data.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The correct answer to compare against.
        extra_info (dict, optional): Additional information for debugging or logging.
    Returns:
        float: The computed score.
    """
    format_score = compute_score_WHWM_format(solution_str, match_score=1.0)
    answer_score = compute_score_answer(solution_str, ground_truth, match_score=2.0)

    return format_score + answer_score


def compute_WHWM_NO_THINK_score(data_source, solution_str, ground_truth, extra_info=None):
    return compute_WHWM_score(data_source, solution_str, ground_truth, extra_info)