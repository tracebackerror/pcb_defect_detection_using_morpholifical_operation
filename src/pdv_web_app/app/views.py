from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.shortcuts import render

from .forms import PCBForm
from .image_ssim import pcb_defect_detection_algorithm
def index(request):
    template = loader.get_template('index.html')


    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = PCBForm(request.POST, request.FILES)
        # check whether it's valid:
        #
        if form.is_valid():
            context = {
                'form': form,
            }
            pcb_defect_detection_algorithm(request.FILES['source_image'], request.FILES['destination_image'])
            return HttpResponse(template.render(context, request))

    else:
        form = PCBForm()

    context = {
        'form': form,
    }

    return HttpResponse(template.render(context, request))
