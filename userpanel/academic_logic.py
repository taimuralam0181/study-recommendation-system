"""
Academic Support Logic (Explainable, Viva-Friendly)

Evaluation scheme (Total = 100):
  Assignment: 5
  Attendance: 5
  Class Test (CT): 10
  Midterm: 30
  Final: 50

Grade cutoffs (department policy):
  Pass: 40
  A-: 70
  A: 75
  A+: 80

Core idea:
  - Compute required final marks to reach a target grade.
  - Detect weak subjects from midterm performance.
  - Generate a structured, realistic recovery plan for final exams.
"""

FINAL_MAX = 50
MIDTERM_MAX = 30

GRADE_CUTOFFS = [
    ("A+", 80),
    ("A", 75),
    ("A-", 70),
    ("B+", 65),
    ("B", 60),
    ("C+", 55),
    ("C", 50),
    ("D", 40),
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def prefinal_total(hand, attendance, ct, midterm):
    return hand + attendance + ct + midterm


def required_final_for_cutoff(prefinal, cutoff):
    # F_req = max(0, cutoff - prefinal)
    return max(0, cutoff - prefinal)


def is_feasible_required_final(f_req):
    return f_req <= FINAL_MAX


def best_feasible_grade(prefinal):
    """
    Returns the highest grade that is mathematically achievable given the current pre-final total.
    """
    t_max = prefinal + FINAL_MAX
    for grade, cutoff in GRADE_CUTOFFS:
        if cutoff <= t_max:
            f_req = required_final_for_cutoff(prefinal, cutoff)
            return {
                "grade": grade,
                "cutoff": cutoff,
                "required_final": f_req,
                "max_total": t_max,
            }
    return {
        "grade": "Fail",
        "cutoff": None,
        "required_final": None,
        "max_total": t_max,
    }


def midterm_band(midterm):
    """
    Midterm is out of 30.
    Weak: <15 (below 50%)
    Average: 15..22 (50%..73%)
    Strong: >=23 (75%+)
    """
    if midterm < 15:
        return "Weak"
    if midterm < 23:
        return "Average"
    return "Strong"


def urgency_tier(required_final):
    """
    Urgency based on required final out of 50.
    High: 45..50 (>=90%)
    Medium: 38..44 (>=75% and <90%)
    Low: 0..37 (<75%)
    """
    pct = (required_final / FINAL_MAX) * 100 if FINAL_MAX else 0
    if pct >= 90:
        return "High"
    if pct >= 75:
        return "Medium"
    return "Low"


def explain_target(prefinal, target_grade, required_final):
    if target_grade == "Fail":
        return "Even passing is not feasible under current constraints."

    if prefinal >= 30:
        aplus_possible = "A+" if target_grade == "A+" else "a high grade"
        return (
            f"Pre-final total is {prefinal}. The final exam contributes up to {FINAL_MAX} marks, "
            f"so {aplus_possible} may be achievable if the required final score is met."
        )
    return (
        f"Pre-final total is {prefinal}. The system selects the highest feasible grade based on the "
        f"maximum possible total (pre-final + final max). Required final: {required_final}/{FINAL_MAX}."
    )


def generate_recovery_plan(midterm, required_final):
    """
    Returns a short, structured plan suitable for dashboards and reports.
    """
    band = midterm_band(midterm)
    tier = urgency_tier(required_final)

    steps = []
    if band == "Weak":
        steps.append("Week 1: Concept repair (revise 2–3 weakest syllabus units + short notes).")
        steps.append("Week 2: Practice ladder (basic → medium → past questions).")
        steps.append("Week 3+: Timed mocks and an error-log; re-practice weak areas.")
    elif band == "Average":
        steps.append("Week 1: Targeted revision of frequently tested topics.")
        steps.append("Week 2: Past papers + structured problem sets.")
        steps.append("Week 3+: Timed mocks focusing on speed and accuracy.")
    else:
        steps.append("Week 1: Consolidation revision + summary sheets.")
        steps.append("Week 2: Timed past papers; refine any weak subtopics.")
        steps.append("Week 3+: Mock exams to secure the target comfortably.")

    if tier == "High":
        steps.append("Urgency: Aim for 45+ in the final; schedule 2 mocks/week.")
    elif tier == "Medium":
        steps.append("Urgency: Aim for 38–44 in the final; schedule 1–2 mocks/week.")
    else:
        steps.append("Urgency: Target is achievable with steady practice; 1 mock/week is sufficient.")

    return band, tier, steps


def grade_from_total(total_score: float) -> str:
    """
    Converts a total score (0..100) into a letter grade using department cutoffs.
    """
    for grade, cutoff in GRADE_CUTOFFS:
        if total_score >= cutoff:
            return grade
    return "F"
