from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userpanel', '0011_studentsubjectperformance_attendance_percentage_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='subject',
            name='department',
            field=models.CharField(default='CSE', max_length=50),
        ),
        migrations.AddField(
            model_name='subject',
            name='is_active',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterUniqueTogether(
            name='subject',
            unique_together={('semester', 'name', 'department')},
        ),
    ]
