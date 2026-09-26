"""Grading prompts copied verbatim from the user-provided HBA repository.
Source: evaluation/llm_grader/llm_judge_eval.py
SHA256: 8beb17ad64b844d648a1dbde2750a87e43c79ad43b546f76f9bf81e963eba5c7
"""

MIMEQA_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answers regarding understanding of a mime performance.
You will be shown a "gold-standard" answer from a human annotator, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to determine whether the candidate captures the core meaning of the reference answer using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate does not contain misleading information and does not hallucinate story plots not present in the reference answer.
3. Since the videos are mime performances, invisible actions, objects, or the mime actor portraying objects should be considered correct if and only if they are relevant to the question.
4. The candidate answer can be a good answer in place of the reference answer as long as they are in the same ballpark. However, the candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

SIQ_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answer regarding understanding of a video.
You will be shown a "gold-standard" answer from human annotators, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to judge whether the candidate captures the core meaning of the reference answer using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate's explanation is semantically equivalent as the reference and does not add a claim that conflicts with it. 
3. The candidate should not assert a conflicting explanation or introduce factually incompatible details. The candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

INTENTQA_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the question and answers about the intent behind actions in a video.
You will be shown a "gold-standard" answer from human annotators, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to judge whether the candidate gives a plausible interpretation of the intent that does not contradict the reference, using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate's explanation is in the same ballpark as the reference and does not add a claim that conflicts with it. The wording need not be the same; minor additions are allowed if they are consistent with the reference and the question.
3. The candidate should not assert a conflicting explanation, introduce factually incompatible details, or miss the core intent. The candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

GRADE_PROMPT = """
Question:
"{question}"
Candidate Answer:
"{candidate_answer}"
Reference Answer:
"{ref_answer}"

Please evaluate the candidate answer based on the dataset-specific instructions.

Respond with exactly this format - a JSON object with two fields:
- "correct": true or false (boolean)
- "explanation": a very short, few phrases explanation of your decision (string)
Only respond with the JSON object, no other text or comments.
"""
