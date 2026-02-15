import argparse
import random
from pathlib import Path

import pandas as pd


SUBJECTS_BY_SEMESTER = {
    1: ["Structured Programming", "Discrete Mathematics", "Calculus"],
    2: ["Data Structures", "Digital Logic Design", "Linear Algebra"],
    3: ["Algorithms", "Database Systems", "Computer Architecture"],
    4: ["Operating Systems", "Software Engineering", "Theory of Computation"],
    5: ["Computer Networks", "Machine Learning", "Compiler Design"],
    6: ["Artificial Intelligence", "Web Engineering", "Information Security"],
    7: ["Data Mining", "Cloud Computing", "Mobile App Development"],
    8: ["Deep Learning", "Distributed Systems", "Project and Thesis"],
}


def clamp(value, low, high):
    return max(low, min(high, value))


def grade_from_total(total):
    if total >= 80:
        return "A+"
    if total >= 75:
        return "A"
    if total >= 70:
        return "A-"
    if total >= 65:
        return "B+"
    if total >= 60:
        return "B"
    if total >= 55:
        return "C+"
    if total >= 50:
        return "C"
    if total >= 40:
        return "D"
    return "F"


def generate_rows(students, seed):
    rng = random.Random(seed)
    rows = []
    for sid in range(1, students + 1):
        cgpa_prev = rng.uniform(2.0, 3.9)
        for semester, subjects in SUBJECTS_BY_SEMESTER.items():
            for subject in subjects:
                assignment = clamp(rng.gauss(3.2 + (cgpa_prev - 3.0), 0.9), 0.0, 5.0)
                quiz = clamp(rng.gauss(6.0 + (cgpa_prev - 3.0) * 1.5, 1.8), 0.0, 10.0)
                attendance_pct = clamp(rng.gauss(82.0 + (cgpa_prev - 3.0) * 8.0, 12.0), 0.0, 100.0)
                midterm = clamp(rng.gauss(18.0 + (cgpa_prev - 3.0) * 5.0, 5.0), 0.0, 30.0)

                attendance_marks = 5.0 * (attendance_pct / 100.0)
                prefinal = assignment + quiz + midterm + attendance_marks
                final = clamp(
                    rng.gauss((prefinal / 50.0) * 38.0 + cgpa_prev * 3.0, 5.0),
                    0.0,
                    50.0,
                )
                total = prefinal + final
                grade = grade_from_total(total)

                rows.append(
                    {
                        "student_id": f"S{sid:05d}",
                        "semester": semester,
                        "subject": subject,
                        "assignment_marks": round(assignment, 2),
                        "quiz_marks": round(quiz, 2),
                        "attendance_percentage": round(attendance_pct, 2),
                        "midterm_marks": round(midterm, 2),
                        "previous_cgpa": round(cgpa_prev, 2),
                        "final_marks": round(final, 2),
                        "final_grade": grade,
                    }
                )

                cgpa_prev = clamp((cgpa_prev * 0.85) + ((total / 25.0) * 0.15), 0.0, 4.0)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate example university ML training dataset.")
    parser.add_argument("--output", default="media/datasets/university_example_dataset.csv")
    parser.add_argument("--students", type=int, default=120, help="Number of students. 120 yields 2880+ rows.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_rows(args.students, args.seed)
    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
