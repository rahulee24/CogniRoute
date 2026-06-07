import pytest
from quality.grader import ContextGrader

def test_context_grader():
    grader = ContextGrader()
    
    # Empty context
    res_empty = grader.grade("refund policy", "")
    assert res_empty["sufficient"] is False
    assert res_empty["score"] == 0.0
    
    # Highly relevant context
    context = "Our billing agreement includes a 14-day refund policy for all plan tiers."
    res_good = grader.grade("refund policy", context)
    assert res_good["sufficient"] is True
    assert res_good["score"] >= 0.7
    
    # Irrelevant context
    context_bad = "The cat sat on the mat."
    res_bad = grader.grade("refund policy", context_bad)
    assert res_bad["sufficient"] is False
