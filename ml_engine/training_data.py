"""
Academic Dataset Generator (Synthetic, Realistic, Explainable)

We generate historical student records where final marks are known (for training only).
In the live system, students input pre-final signals; we predict final marks and grade probabilities.

Why synthetic is acceptable academically:
- Real student data is sensitive (privacy); synthetic avoids unethical data usage.
- We explicitly encode realistic relationships:
  - attendance and assignments correlate with better finals
  - midterm is predictive but not perfect
  - previous CGPA influences performance
  - subject difficulty introduces noise
"""

from __future__ import annotations

from typing import Dict, List
import random


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_examples(
    n_students: int,
    semesters: List[int],
    subjects_by_semester: Dict[int, List[str]],
    seed: int = 42,
) -> List[Dict]:
    rnd = random.Random(seed)
    rows: List[Dict] = []

    # Student latent ability ~ N-like via sum of uniforms (0..1)
    def ability():
        return (rnd.random() + rnd.random() + rnd.random()) / 3.0

    for sid in range(1, n_students + 1):
        # Previous CGPA (0..4) starts from ability
        cgpa_prev = clamp(1.5 + 2.5 * ability(), 0.0, 4.0)

        for sem in semesters:
            for subj in subjects_by_semester[sem]:
                # Subject difficulty 0..1 (higher means harder)
                difficulty = rnd.random()

                # Components: generated with correlation to cgpa_prev and ability
                # assignment_marks (0..5), attendance_percentage (0..100), quiz_marks (0..10), midterm_marks (0..30)
                base = 0.55 * (cgpa_prev / 4.0) + 0.45 * ability()

                attendance_percentage = clamp(100 * (0.65 * base + 0.35 * rnd.random()), 0, 100)
                attendance_marks = clamp(5 * (attendance_percentage / 100.0), 0, 5)
                assignment_marks = clamp(5 * (0.65 * base + 0.35 * rnd.random()), 0, 5)
                quiz_marks = clamp(10 * (0.6 * base + 0.4 * rnd.random()), 0, 10)

                # Midterm is affected by difficulty
                midterm_raw = 30 * (0.65 * base + 0.35 * rnd.random()) - (difficulty * 6)
                midterm_marks = clamp(midterm_raw, 0, 30)

                # Final marks (target) are influenced by midterm + effort signals + cgpa_prev + noise
                # Also allow "recovery": a low midterm student can still do well if effort signals are strong.
                effort = (assignment_marks / 5.0) * 0.4 + (attendance_marks / 5.0) * 0.2 + (quiz_marks / 10.0) * 0.4
                recovery_boost = 6.0 * max(0.0, effort - (midterm_marks / 30.0))  # recovery effect

                final_raw = (
                    0.45 * (midterm_marks / 30.0) +
                    0.35 * effort +
                    0.20 * (cgpa_prev / 4.0)
                ) * 50.0 + recovery_boost - (difficulty * 5.0) + rnd.uniform(-4.0, 4.0)

                final_marks = clamp(final_raw, 0, 50)

                total = clamp(assignment_marks + attendance_marks + quiz_marks + midterm_marks + final_marks, 0, 100)
                # Update cgpa_prev gradually by semester performance
                cgpa_prev = clamp(0.85 * cgpa_prev + 0.15 * (total / 25.0), 0.0, 4.0)

                def grade(t):
                    if t >= 80:
                        return "A+"
                    if t >= 75:
                        return "A"
                    if t >= 70:
                        return "A-"
                    if t >= 65:
                        return "B+"
                    if t >= 60:
                        return "B"
                    if t >= 55:
                        return "C+"
                    if t >= 50:
                        return "C"
                    if t >= 40:
                        return "D"
                    return "F"

                rows.append({
                    "student_id": sid,
                    "semester": sem,
                    "subject": subj,
                    "assignment_marks": round(assignment_marks, 2),
                    "attendance_percentage": round(attendance_percentage, 2),
                    "quiz_marks": round(quiz_marks, 2),
                    "midterm_marks": round(midterm_marks, 2),
                    "previous_cgpa": round(cgpa_prev, 2),
                    "final_marks": round(final_marks, 2),  # training target (NOT user input)
                    "final_grade": grade(total),
                    "total": round(total, 2),
                })

    return rows
