from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme

from accounts.models import UserProfile
from userpanel.models import Student


def _get_role(user) -> str:
    profile, _ = UserProfile.objects.get_or_create(user=user, defaults={"role": "student"})
    return profile.role


def login_view(request):
    # If already logged in, go to the correct dashboard.
    if request.user.is_authenticated:
        return redirect("teacher_dashboard" if _get_role(request.user) == "teacher" else "dashboard")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        next_url = (request.POST.get("next") or "").strip()

        user = authenticate(request, username=username, password=password)
        if user is None:
            return render(
                request,
                "accounts/login.html",
                {"error": "Invalid username or password.", "next": next_url, "username": username},
            )

        login(request, user)
        role = _get_role(user)

        # Respect "next" for a smoother UX, but do not allow students to enter teacher routes.
        if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
            if not (role == "student" and next_url.startswith("/teacher/")):
                return redirect(next_url)

        return redirect("teacher_dashboard" if role == "teacher" else "dashboard")

    return render(request, "accounts/login.html", {"next": (request.GET.get("next") or "").strip()})


def register_view(request):
    if request.user.is_authenticated:
        return redirect("teacher_dashboard" if _get_role(request.user) == "teacher" else "dashboard")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        confirm = request.POST.get("confirm_password", "")
        role = request.POST.get("role", "student")

        if not username:
            return render(request, "accounts/register.html", {"error": "Username is required."})
        if len(password) < 6:
            return render(request, "accounts/register.html", {"error": "Password must be at least 6 characters."})
        if password != confirm:
            return render(request, "accounts/register.html", {"error": "Passwords do not match."})
        if User.objects.filter(username=username).exists():
            return render(request, "accounts/register.html", {"error": "Username already exists."})

        role_clean = role if role in ["teacher", "student"] else "student"

        user = User.objects.create_user(username=username, password=password)
        UserProfile.objects.create(user=user, role=role_clean)

        # ML inference uses previous CGPA; keep a Student row for student accounts.
        if role_clean == "student":
            Student.objects.get_or_create(user=user, defaults={"cgpa": 0.0})

        login(request, user)
        messages.success(request, "Account created successfully.")
        return redirect("teacher_dashboard" if role_clean == "teacher" else "dashboard")

    return render(request, "accounts/register.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required
def dashboard_view(request):
    # Keep /dashboard/ route unchanged and delegate UI to userpanel.
    role = _get_role(request.user)
    if role == "teacher":
        return redirect("teacher_dashboard")

    from userpanel.views import dashboard as student_dashboard

    return student_dashboard(request)


@login_required
def cyberpunk_dashboard(request):
    """
    Cyberpunk-themed React Native for Web Dashboard
    Serves the futuristic UI dashboard with neon effects
    """
    role = _get_role(request.user)
    if role == "teacher":
        return redirect("teacher_dashboard")
    
    return render(request, "dashboard/index.html")
