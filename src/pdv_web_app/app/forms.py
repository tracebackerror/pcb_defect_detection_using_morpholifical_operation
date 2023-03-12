from django import forms
from string import Template
from django.utils.safestring import mark_safe
from django.forms import ImageField

class PictureWidget(forms.widgets.Widget):
    def render(self, name, value, attrs=None, **kwargs):
        html =  Template("""<img src="$link"/>""")
        return mark_safe(html.substitute(link=value))

class PCBForm(forms.Form):
    source_image = forms.ImageField(label='First PCB',)
    destination_image = forms.ImageField(label='Second PCB', )
