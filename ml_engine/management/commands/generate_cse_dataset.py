from django.core.management.base import BaseCommand

from userpanel.models import Semester, Subject
from ml_engine.models import CSETrainingExample
from ml_engine.training_data import generate_examples


class Command(BaseCommand):
    """
    Generates a realistic synthetic academic dataset for CSE (Semester 1..8).

    Students do NOT input final marks in the live system.
    We include final_marks here ONLY to train/evaluate ML models.
    """

    def add_arguments(self, parser):
        parser.add_argument('--students', type=int, default=300)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--clear', action='store_true')

    def handle(self, *args, **opts):
        n = opts['students']
        seed = opts['seed']
        clear = opts['clear']

        if clear:
            CSETrainingExample.objects.all().delete()

        semesters = list(Semester.objects.order_by('number').values_list('number', flat=True))
        if not semesters:
            self.stdout.write(self.style.ERROR("No semesters found. Run seed_cs_program first."))
            return

        subjects_by_sem = {}
        for sem_no in semesters:
            subjects_by_sem[sem_no] = list(
                Subject.objects.filter(semester__number=sem_no).order_by('name').values_list('name', flat=True)
            )
            if len(subjects_by_sem[sem_no]) == 0:
                self.stdout.write(self.style.ERROR(f"No subjects found for semester {sem_no}."))
                return

        rows = generate_examples(
            n_students=n,
            semesters=semesters,
            subjects_by_semester=subjects_by_sem,
            seed=seed,
        )

        objs = [
            CSETrainingExample(
                student_id=str(r["student_id"]),
                semester=r["semester"],
                subject=r["subject"],
                assignment_marks=r["assignment_marks"],
                attendance_percentage=r["attendance_percentage"],
                quiz_marks=r["quiz_marks"],
                midterm_marks=r["midterm_marks"],
                previous_cgpa=r["previous_cgpa"],
                final_marks=r["final_marks"],
                total=r["total"],
                final_grade=r["final_grade"],
            )
            for r in rows
        ]

        CSETrainingExample.objects.bulk_create(objs, batch_size=2000)
        self.stdout.write(self.style.SUCCESS(f"Generated {len(objs)} training rows for {n} students."))
