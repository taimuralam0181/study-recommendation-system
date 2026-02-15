from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


def clear_study_materials(apps, schema_editor):
    StudyMaterial = apps.get_model('ml_engine', 'StudyMaterial')
    StudyMaterial.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('userpanel', '0012_subject_department_is_active'),
        ('ml_engine', '0009_remove_csetrainingexample_attendance_and_more'),
    ]

    operations = [
        migrations.RunPython(clear_study_materials, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name='studymaterial',
            name='semester',
        ),
        migrations.RemoveField(
            model_name='studymaterial',
            name='level',
        ),
        migrations.RemoveField(
            model_name='studymaterial',
            name='source',
        ),
        migrations.RemoveField(
            model_name='studymaterial',
            name='topic',
        ),
        migrations.AlterField(
            model_name='studymaterial',
            name='subject',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='userpanel.subject'),
        ),
        migrations.AlterField(
            model_name='studymaterial',
            name='material_type',
            field=models.CharField(choices=[('PDF', 'PDF'), ('Image', 'Image'), ('Link', 'Link')], max_length=30),
        ),
        migrations.AddField(
            model_name='studymaterial',
            name='file',
            field=models.FileField(blank=True, upload_to='materials/'),
        ),
        migrations.AddField(
            model_name='studymaterial',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
