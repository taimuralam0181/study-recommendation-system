from django import forms

from ml_engine.models import UploadedDataset


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedDataset
        fields = ["file"]

    def clean_file(self):
        file = self.cleaned_data["file"]
        max_size = 10 * 1024 * 1024
        if file.size and file.size > max_size:
            raise forms.ValidationError("File size must be 10MB or less.")
        content_type = (file.content_type or "").lower()
        allowed_types = {
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        if content_type and content_type not in allowed_types:
            raise forms.ValidationError("Only CSV or Excel uploads are allowed.")
        if not file.name.lower().endswith((".csv", ".xlsx", ".xls")):
            raise forms.ValidationError("Please upload a CSV or Excel file.")
        return file
