from accounts.models import UserProfile


def current_role(request):
    role = None
    user = getattr(request, "user", None)
    if user and user.is_authenticated:
        profile, _ = UserProfile.objects.get_or_create(user=user, defaults={"role": "student"})
        role = profile.role
    return {"current_role": role}
