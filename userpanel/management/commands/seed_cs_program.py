import random
from urllib.parse import quote_plus

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

from accounts.models import UserProfile
from userpanel.models import Semester, Subject, StudentSubjectPerformance, Student
from ml_engine.models import StudyMaterial
from ml_engine.ml_model import recommend_level


class Command(BaseCommand):
    """
    Seeds an 8-semester CS program with:
    - 23 subjects per semester (core CS-oriented list)
    - Synthetic student performance (Hand 20, Attendance 5, CT 15, Midterm 30, Final 50 -> Total 100)
    - Syllabus-aligned study materials (MIT OCW / Stanford / MDN / NPTEL links)

    This is designed for academic demos: predictable, explainable, and reproducible.
    """

    def add_arguments(self, parser):
        parser.add_argument('--students', type=int, default=30)
        parser.add_argument('--clear', action='store_true')
        parser.add_argument('--seed', type=int, default=42)

    def handle(self, *args, **opts):
        n_students = opts['students']
        seed = opts['seed']
        clear = opts['clear']

        random.seed(seed)

        if clear:
            StudentSubjectPerformance.objects.all().delete()
            Subject.objects.all().delete()
            Semester.objects.all().delete()
            StudyMaterial.objects.all().delete()

        syllabus = self._syllabus()

        # Create semesters + subjects
        for sem_no in range(1, 9):
            sem, _ = Semester.objects.get_or_create(number=sem_no, defaults={'title': f'CS Semester {sem_no}'})
            for name in syllabus[sem_no]:
                Subject.objects.get_or_create(
                    semester=sem,
                    name=name,
                    defaults={'department': 'CSE', 'is_active': True},
                )

        # Create materials (1 link per subject, minimal but syllabus-aligned)
        for subj in Subject.objects.select_related('semester').all():
            q = quote_plus(subj.name)
            sem_no = subj.semester.number

            # Use simple, trusted search pages (stable) rather than brittle deep links.
            mit = f"https://ocw.mit.edu/search/?q={q}"
            mdn = f"https://developer.mozilla.org/en-US/search?q={q}"

            # Heuristic: web-heavy subjects prefer MDN as a primary source.
            webish = any(k in subj.name.lower() for k in ['web', 'html', 'css', 'javascript', 'frontend', 'backend'])

            link = mdn if webish else mit

            StudyMaterial.objects.get_or_create(
                subject=subj,
                title=f"{subj.name} Resources",
                defaults={
                    'description': f"Curated resources aligned to {subj.name} (Semester {sem_no}).",
                    'material_type': 'Link',
                    'link': link,
                    'added_by': self._ensure_teacher_user(),
                }
            )

        # Create synthetic students and performance
        for i in range(1, n_students + 1):
            username = f"student{i:03d}"
            user, created = User.objects.get_or_create(username=username)
            if created:
                user.set_password("student123")
                user.save()
            UserProfile.objects.get_or_create(user=user, defaults={'role': 'student'})

            # Assign a performance band for realism
            band = random.choice(['low', 'mid', 'high'])
            band_ranges = {
                'low': {
                    'hand': (0, 3), 'att': (2, 5), 'ct': (1, 6), 'mid': (4, 15), 'final': (10, 35)
                },
                'mid': {
                    'hand': (2, 5), 'att': (3, 5), 'ct': (4, 9), 'mid': (12, 23), 'final': (25, 45)
                },
                'high': {
                    'hand': (4, 5), 'att': (4, 5), 'ct': (8, 10), 'mid': (20, 30), 'final': (35, 50)
                },
            }
            rr = band_ranges[band]

            for subj in Subject.objects.select_related('semester').all():
                hand = random.randint(*rr['hand'])
                att = random.randint(*rr['att'])
                ct = random.randint(*rr['ct'])
                mid = random.randint(*rr['mid'])
                fin = random.randint(*rr['final'])
                total = hand + att + ct + mid + fin
                if total > 100:
                    # Keep totals within the academic maximum (100).
                    total = 100

                StudentSubjectPerformance.objects.update_or_create(
                    user=user,
                    semester=subj.semester,
                    subject=subj,
                    defaults={
                        'hand_marks': hand,
                        'attendance_marks': att,
                        'ct_marks': ct,
                        'midterm_marks': mid,
                        'final_marks': fin,
                        'total': total,
                    }
                )

            # Keep existing 3-subject Student summary in sync for dashboards/ML demo.
            # We map to the first three subjects from Semester 1 for a consistent baseline.
            s1 = Semester.objects.get(number=1)
            s1_subjects = list(Subject.objects.filter(semester=s1).order_by('name')[:3])
            pick_totals = []
            for s in s1_subjects:
                p = StudentSubjectPerformance.objects.get(user=user, semester=s1, subject=s)
                pick_totals.append(p.total)

            math, physics, cs = (pick_totals + [0, 0, 0])[:3]
            level = recommend_level(math, physics, cs)
            avg_score = (math + physics + cs) / 3

            Student.objects.update_or_create(
                user=user,
                defaults={
                    'math_marks': math,
                    'physics_marks': physics,
                    'cs_marks': cs,
                    'avg_score': avg_score,
                    'level': level,
                    'cgpa': self._compute_cgpa(user),
                }
            )

        self.stdout.write(self.style.SUCCESS("Seed complete."))
        self.stdout.write("Teacher login: teacher / teacher123")
        self.stdout.write("Sample student login: student001 / student123")

    def _ensure_teacher_user(self):
        user, created = User.objects.get_or_create(username='teacher')
        if created:
            user.set_password('teacher123')
            user.is_staff = True
            user.save()
        UserProfile.objects.get_or_create(user=user, defaults={'role': 'teacher'})
        return user

    def _compute_cgpa(self, user):
        """
        Explainable CGPA approximation:
        - Convert each subject total (0..100) to a grade point (0.0..4.0)
        - CGPA is the mean grade point across all subjects.
        """
        totals = list(
            StudentSubjectPerformance.objects.filter(user=user).values_list('total', flat=True)
        )
        if not totals:
            return 0.0

        def gp(t):
            if t >= 80:
                return 4.0
            if t >= 75:
                return 3.75
            if t >= 70:
                return 3.5
            if t >= 60:
                return 3.0
            if t >= 50:
                return 2.5
            if t >= 40:
                return 2.0
            return 0.0

        points = [gp(t) for t in totals]
        return round(sum(points) / len(points), 2)

    def _syllabus(self):
        # 23 core CS-oriented subjects per semester (university-grade list).
        return {
            1: [
                "Programming Fundamentals",
                "Discrete Mathematics",
                "Introduction to Computing Systems",
                "Digital Logic Design",
                "Calculus for Computing",
                "Communication Skills",
                "Applied Physics for Computing",
                "Basic Electrical Engineering",
                "Problem Solving Techniques",
                "Computer Lab I",
                "Mathematics for CS I",
                "Introduction to Data",
                "Ethics in Computing",
                "Linear Algebra Basics",
                "Academic Writing",
                "Open Source Tools",
                "IT Fundamentals",
                "C Programming",
                "Technical Presentation",
                "Critical Thinking",
                "Innovation & Entrepreneurship",
                "Environmental Studies",
                "Seminar I",
            ],
            2: [
                "Object Oriented Programming",
                "Data Structures",
                "Computer Organization",
                "Probability and Statistics",
                "Mathematics for CS II",
                "Database Fundamentals",
                "Operating Systems Basics",
                "Web Technologies I",
                "Software Engineering Fundamentals",
                "Computer Lab II",
                "Design and Analysis of Algorithms I",
                "Signals and Systems for CS",
                "Technical Writing",
                "Cybersecurity Basics",
                "Human Computer Interaction",
                "Numerical Methods",
                "Python Programming",
                "Communication Networks Basics",
                "Professional Ethics",
                "Project I",
                "Seminar II",
                "Economics for Engineers",
                "Industry Orientation",
            ],
            3: [
                "Design and Analysis of Algorithms II",
                "Operating Systems",
                "Database Management Systems",
                "Computer Networks",
                "Theory of Computation",
                "Microprocessors and Interfacing",
                "Software Engineering",
                "Web Technologies II",
                "Linear Algebra for ML",
                "Data Communication",
                "Object Oriented Analysis and Design",
                "Unix and Shell Programming",
                "Mobile Application Basics",
                "Cloud Fundamentals",
                "Computer Graphics Basics",
                "Digital Signal Processing Basics",
                "Technical Seminar III",
                "Project II",
                "Professional Communication",
                "Open Elective I (CS)",
                "Lab: OS/Networks",
                "Lab: DBMS",
                "Lab: Web Development",
            ],
            4: [
                "Compiler Design",
                "Artificial Intelligence",
                "Machine Learning Fundamentals",
                "Information Security",
                "Distributed Systems",
                "Data Mining",
                "Computer Graphics",
                "Advanced Database Systems",
                "Software Testing",
                "Internet of Things",
                "Design Patterns",
                "Agile Development",
                "Big Data Fundamentals",
                "Research Methodology",
                "Professional Seminar IV",
                "Project III",
                "Lab: AI/ML",
                "Lab: Security",
                "Lab: Distributed Systems",
                "Open Elective II (CS)",
                "Numerical Computing",
                "Technical Documentation",
                "Industry Case Studies",
            ],
            5: [
                "Deep Learning (Concepts)",
                "Natural Language Processing",
                "Computer Vision",
                "Advanced Networks",
                "Cloud Computing",
                "DevOps Fundamentals",
                "Blockchain Basics",
                "Advanced Operating Systems",
                "Advanced Algorithms",
                "Software Project Management",
                "Data Warehousing",
                "Human Centered Computing",
                "Elective III (CS)",
                "Elective IV (CS)",
                "Lab: Cloud/DevOps",
                "Lab: NLP/CV",
                "Project IV",
                "Seminar V",
                "Research Paper Review",
                "Engineering Management",
                "Technical Interview Prep",
                "Ethics and Law in Tech",
                "Mini Project",
            ],
            6: [
                "MLOps Basics",
                "Advanced Data Science",
                "Parallel and Distributed Computing",
                "Advanced Security",
                "Information Retrieval",
                "Recommender Systems",
                "Advanced Web Development",
                "Scalable Systems",
                "Elective V (CS)",
                "Elective VI (CS)",
                "Lab: Systems",
                "Lab: Data Science",
                "Project V",
                "Seminar VI",
                "Research Proposal",
                "Software Architecture",
                "Quality Assurance",
                "Product Design",
                "Cloud Native",
                "Data Privacy",
                "Technical Leadership",
                "Internship Prep",
                "Community Project",
            ],
            7: [
                "Major Project Phase I",
                "Industrial Training / Internship",
                "Seminar VII",
                "Elective VII (CS)",
                "Elective VIII (CS)",
                "Advanced Topics in AI",
                "Advanced Topics in Systems",
                "Research Writing",
                "Entrepreneurship in CS",
                "Technology Policy",
                "Project Management Tools",
                "System Design Workshop",
                "Security Audit Workshop",
                "Cloud Workshop",
                "Data Engineering Workshop",
                "Hackathon Practicum",
                "Open Source Contribution",
                "Professional Development",
                "Ethics Seminar",
                "Innovation Lab",
                "Capstone Planning",
                "Technical Seminar",
                "Placement Preparation",
            ],
            8: [
                "Major Project Phase II",
                "Seminar VIII",
                "Comprehensive Viva",
                "Thesis / Dissertation",
                "Elective IX (CS)",
                "Elective X (CS)",
                "Advanced Research Topics",
                "Startup Incubation",
                "Professional Ethics (Advanced)",
                "Project Demo and Report",
                "Industry Seminar",
                "Publication Workshop",
                "Patent Basics",
                "Technology Transfer",
                "Leadership and Communication",
                "Career Planning",
                "System Integration",
                "Testing and Validation",
                "Deployment and Monitoring",
                "Documentation and Standards",
                "Capstone Presentation",
                "Final Review",
                "Exit Survey",
            ],
        }
